# Snece Classification (DSA5203 project)

Our empirical studies：

Model1: Vanilla CNN with 3 layers. 

Model2: modified(simplified) VGG [[paper](https://arxiv.org/abs/1409.1556)].

Model3: (simplified) ResNet [[paper](https://arxiv.org/abs/1512.03385)].

Model4: Attention Residual Network [[paper](https://arxiv.org/abs/1704.06904)].

Modified VGG achieved the best result on scene dataset, with accuracy of up to 79% on the validation set. 


## Experiment Results

![image](https://github.com/h1226933031/scene-classification/blob/main/illustration/experiment%20results.png)


## Get Started

1. Install Python>=3.8, PyTorch 1.12.1.
2. Download scene image data and codes from this repo.
3. Train the model: Directly run the `./run.py` or run `bash ./run.sh` to get the model trained under specified training settings.

```bash
bash ./run.sh
```


## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/weiaicunzai/pytorch-cifar100

