
# Diff-UNet

Diff-UNet: A Diffusion Embedded Network for Volumetric Segmentation.

We design the Diff-UNet applying diffusion model to solve the 3D medical image segmentation problem.

Diff-UNet achieves more accuracy in multiple segmentation tasks compared with other 3D segmentation methods.

![](/imgs/framework.png)

## dataset 
We release the codes which support the training and testing process of two dataset, BraTS2020 and BTCV.

BraTS2020(4 modalities and 3 segmentation targets): https://www.med.upenn.edu/cbica/brats2020/data.html

BTCV(1 modalities and 13 segmentation targets): https://www.synapse.org/#!Synapse:syn3193805/wiki/217789

Once the data is downloaded, you can begin the training process.

## training 

Training use Pytoch DDP mode with four GPUs. You also can modify the parameters to use one GPU to train(refer to the train.py).

```bash
python train.py
```

## testing
When you have trained a model, please modify the model path, then run the code.
```bash
python test.py
```