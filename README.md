# Snece Classification (DSA5203 project)

Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, Rong Jin, "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting," in Proc. 39th International Conference on Machine Learning (ICML 2022), Baltimore, Maryland, July 17-23, 2022

Model1: Vanilla CNN with 3 layers. 
Model2: modified(simplified) VGG [[paper](https://arxiv.org/abs/1409.1556)].
Model3: (simplified) ResNet [[paper](https://arxiv.org/abs/1512.03385)].
Model4: Attention Residual Network [[paper](https://arxiv.org/abs/1704.06904)].

Our empirical studies
Modified VGG achieved the best result on scene dataset, with accuracy of up to 79% on the validation set. 
with six benchmark datasets show that compared
with state-of-the-art methods, FEDformer can
reduce prediction error by 14.8% and 22.6%
for multivariate and univariate time series,
respectively.


## Main Results
![image](https://user-images.githubusercontent.com/44238026/171345192-e7440898-4019-4051-86e0-681d1a28d630.png)


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

