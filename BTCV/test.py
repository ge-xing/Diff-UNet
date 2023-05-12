import numpy as np
from dataset.btcv_transunet_datasetings import get_loader_btcv
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.trainer import Trainer
from monai.utils import set_determinism
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
from medpy import metric

set_determinism(123)

max_epoch = 300
batch_size = 2
val_every = 10
num_gpus = 2
device = "cuda:0"

data_dir = "./RawData/Training/"


def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.01] = 0.01
    uncer_out = - pred_out * torch.log(pred_out)

    return uncer_out

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
            return self.model(x, t=step, image=image, embedding=embedding)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, 13, 96, 96, 96), model_kwargs={"image": image, "embeddings": embeddings})

            sample_return = torch.zeros((1, 13, 96, 96, 96))
            all_samples = sample_out["all_samples"]
            index = 0
            for sample in all_samples:
                sample_return += sample.cpu()
                index += 1

            return sample_return


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                        sw_batch_size=1,
                                        overlap=0.6)
        
        self.model = DiffUNet()

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)

        self.loss_func = nn.CrossEntropyLoss()

    def get_input(self, batch):
        image = batch["image"]
        label = batch["raw_label"]
        
        label = self.convert_labels(label)

        label = label.float()
        return image, label 

    def convert_labels(self, labels):
        labels_new = []
        for i in range(1, 14):
            labels_new.append(labels == i)
        
        labels_new = torch.cat(labels_new, dim=1)
        return labels_new

    def validation_step(self, batch):
        image, label = self.get_input(batch)    
        
        output = self.window_infer(image, self.model, pred_type="ddim_sample")
        output = torch.sigmoid(output)
        output = (output > 0.5).float().cpu()

        d, w, h = label.shape[2], label.shape[3], label.shape[4]

        output = torch.nn.functional.interpolate(output, mode="nearest", size=(d, w, h))
        output = output.numpy()

        target = label.cpu().numpy()

        dices = []
        hd = []
        c = 13
        for i in range(0, c):
            pred = output[:, i]
            gt = target[:, i]

            if pred.sum() > 0 and gt.sum()>0:
                dice = metric.binary.dc(pred, gt)
                hd95 = metric.binary.hd95(pred, gt)
            elif pred.sum() > 0 and gt.sum()==0:
                dice = 1
                hd95 = 0
            else:
                dice = 0
                hd95 = 0

            dices.append(dice)
            hd.append(hd95)
        
        all_m = []
        for d in dices:
            all_m.append(d)
        for h in hd:
            all_m.append(h)
        print(all_m)
        return all_m 

if __name__ == "__main__":

    train_ds, val_ds, test_ds = get_loader_btcv(data_dir=data_dir)
    
    trainer = BraTSTrainer(env_type="pytorch",
                                    max_epochs=max_epoch,
                                    batch_size=batch_size,
                                    device=device,
                                    val_every=val_every,
                                    num_gpus=1,
                                    master_port=17751,
                                    training_script=__file__)

    logdir = "./logs_btcv/diffusion_seg_transunet_datasettings/model/final_model_0.7904.pt"
    trainer.load_state_dict(logdir)
    v_mean, v_out = trainer.validation_single_gpu(val_dataset=test_ds)

    print(f"v_mean is {v_mean}")