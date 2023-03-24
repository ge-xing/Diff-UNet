import numpy as np
from dataset.btcv_transunet_datasetings import get_loader_btcv
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice, hausdorff_distance_95
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
import argparse
from monai.losses.dice import DiceLoss
import yaml
from unet.basic_unet import BasicUNetEncoder
from unet.basic_unet_denose import BasicUNetDe
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
set_determinism(123)
import os

data_dir = "./RawData/Training/"

logdir = "./logs_btcv/diffunet_transunet_datasettings/"
model_save_path = os.path.join(logdir, "model")

max_epoch = 3000
batch_size = 1
val_every = 100
env = "DDP"
num_gpus = 4
# or
# env = "pytorch"
# num_gpus = 1

device = "cuda:0"

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, 1, 2, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, 14, 13, [64, 64, 128, 256, 512, 64], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(1000)


    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, 13, 96, 96, 96), model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                        sw_batch_size=1,
                                        overlap=0.5)
        self.model = DiffUNet()

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-3)
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                  warmup_epochs=100,
                                                  max_epochs=max_epochs)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def training_step(self, batch):
        image, label = self.get_input(batch)
        x_start = label

        x_start = (x_start) * 2 - 1
        x_t, t, noise = self.model(x=x_start, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")

        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss, step=self.global_step)

        return loss 
 
    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]
        
        label = self.convert_labels(label)

        label = label.float()
        return image, label 

    def convert_labels(self, labels):
        labels_new = []
        for i in range(1, 14):
            labels_new.append(labels == i)
        
        labels_new = torch.cat(labels_new, dim=1)
        return labels_new

    def validation_end(self, mean_val_outputs):
        dices = mean_val_outputs
        print(dices)
        mean_dice = sum(dices) / len(dices)

        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")

        print(f" mean_dice is {mean_dice}")

    def validation_step(self, batch):
        image, label = self.get_input(batch)    
        
        output = self.window_infer(image, self.model, pred_type="ddim_sample")

        output = torch.sigmoid(output)

        output = (output > 0.5).float().cpu().numpy()

        target = label.cpu().numpy()
        dices = []
        hd = []
        c = 13
        for i in range(0, c):
            pred_c = output[:, i]
            target_c = target[:, i]

            dices.append(dice(pred_c, target_c))
            hd.append(hausdorff_distance_95(pred_c, target_c))
        
        return dices

    
if __name__ == "__main__":

    trainer = BraTSTrainer(env_type=env,
                            max_epochs=max_epoch,
                            batch_size=batch_size,
                            device=device,
                            logdir=logdir,
                            val_every=val_every,
                            num_gpus=num_gpus,
                            master_port=17751,
                            training_script=__file__)

    train_ds, val_ds, test_ds = get_loader_btcv(data_dir=data_dir)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
