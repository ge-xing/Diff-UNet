

## dataset 
The dataset is BTCV. You can find it and see the details in https://www.synapse.org/#!Synapse:syn3193805/wiki/217789

Once the data is downloaded, you can begin the training process. The data dir is:

```python
data_dir = "./RawData/Training/"
```

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