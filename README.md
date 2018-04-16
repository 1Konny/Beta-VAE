# β-VAE
Pytorch implementation of β-VAE proposed in [this paper]
<br>

### Dependencies
```
python 3.6.4
pytorch 0.3.1.post2
visdom
```
<br>

### Usage
initialize visdom
```
python -m visdom.server -p 55558
```
you can run codes using sh files
```
e.g.
sh run_celeba.sh
sh run_3dchairs.sh
```
or you can run your own experiments by setting parameters manually
```
e.g.
python main.py --beta 4 --lr 1e-4 --z_dim 32 ...
```
check training process on the visdom server
```
localhost:55558
```
<br>

### Results
soon

### Reference
1. β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK

[this paper]: https://openreview.net/pdf?id=Sy2fzU9gl
