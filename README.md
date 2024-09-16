# Pytorch implementation of Maximum Manifold Capacity Representations Loss
> This is not an official implementation from the authors.
> [Official implementation from the authors](https://github.com/ThomasYerxa/mmcr).


Maximum Manifold Capacity Representation Loss (MMCR Loss) is a novel objective function for self-supervised learning (SSL) proposed by researchers in Center for Neural Science, NYU.

This repository aims to offer a convenient MMCR loss module for PyTorch, which can be easily integrated into your projects using `git clone` or `pip install`.

## How to install
```sh
pip3 install mmcr
```
or 
```sh
git clone https://github.com/skyil7/mmcr
cd mmcr
pip install -e .
```
## Usage
```python
import torch
from mmcr import MMCRLoss

loss = MMCRLoss()

input_tensor = torch.randn((8, 16, 128))  # batch_size, n_aug, feature_dim
loss_val = loss(input_tensor)

print(loss_val)
```

## How it works
$$ \mathcal L = \lambda\frac{\sum^N_{i=1}||z_i||_*}{N} - ||C||_*$$
Where $\lambda$ is a trade-off parameter, $||z_i||_*$ is local nuclear norm of the $i$-th sample's augmented matrix, and $||C||_*$ is the global nuclear norm of centroid matrix $C$.

### Arguments
- `lmbda`: Trade-off parameter $\lambda$. default is 0.
- `n_aug`: number of augmented views. If your input tensor is 3-dimensional $(N, k, d)$, you don't need to specify it.

## Original Implementation from the author
- This repository was developed with reference to the [official implementation](https://github.com/ThomasYerxa/mmcr) provided by the authors.