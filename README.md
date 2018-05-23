# β-VAE
Pytorch reproduction of two papers below:
1. [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework, Higgins et al., ICLR, 2017]
2. [Understanding disentangling in β-VAE, Burgess et al., arxiv:1804.03599, 2018]
<br>

### Dependencies
```
python 3.6.4
pytorch 0.3.1.post2
visdom
```
<br>

### Datasets
[same with here]
<br>

### Usage
initialize visdom
```
python -m visdom.server
```
you can reproduce results below by
```
sh run_celeba_H_beta10_z10.sh
sh run_celeba_H_beta10_z32.sh
sh run_3dchairs_H_beta4_z10.sh
sh run_3dchairs_H_beta4_z16.sh
sh run_dsprites_B_gamma100_z10.sh
```
or you can run your own experiments by setting parameters manually.<br>
for objective and model arguments, you have two options H and B indicating methods proposed in Higgins et al. and Burgess et al., respectively.<br>
arguments ```--C_max``` and ```--C_stop_iter``` should be set when ```--objective B```. for further details, please refer to Burgess et al.
```
e.g.
python main.py --dataset 3DChairs --beta 4 --lr 1e-4 --z_dim 10 --objective H --model H --max_iter 1e6 ...
python main.py --dataset dsprites --gamma 1000 --C_max 25 --C_stop_iter 1e5 --lr 5e-4 --z_dim 10 --objective B --model B --max_iter 1e6 ...
```
check training process on the visdom server
```
localhost:8097
```
<br>

### Results
#### 3D Chairs
```
sh run_3dchairs_H_beta4_z10.sh
```
![3dchairs_beta4_z16](misc/3dchairs_H_beta4_z10_traverse.png)
```
sh run_3dchairs_H_beta4_z16.sh
```
![3dchairs_beta4_z16](misc/3dchairs_H_beta4_z16_traverse.png)
#### CelebA
```
sh run_celeba_H_beta10_z10.sh
```
![celeba](misc/celeba_H_beta10_z10_traverse.png)
```
sh run_celeba_H_beta10_z32.sh
```
![celeba](misc/celeba_H_beta10_z32_traverse.png)
#### dSprites
```
sh run_dsprites_B.sh
```
##### visdom line plot
![dsprites_plot](misc/dsprites_plot.png)
##### latent traversal gif(```--save_output True```)
<p align="center">
<img src=misc/dsprites_traverse_ellipse.gif>
<img src=misc/dsprites_traverse_heart.gif>
<img src=misc/dsprites_traverse_random.gif>
</p>
##### reconstruction(left: true, right: reconstruction)
<p align="center">
<img src=misc/dsprites_reconstruction.jpg>
</p>


### Reference
1. [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework, Higgins et al., ICLR, 2017]
2. [Understanding disentangling in β-VAE, Burgess et al., arxiv:1804.03599, 2018]
3. [Github Repo]: Tensorflow implementation from [miyosuda]

[β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework, Higgins et al., ICLR, 2017]: https://openreview.net/pdf?id=Sy2fzU9gl
[Understanding disentangling in β-VAE, Burgess et al., arxiv:1804.03599, 2018]: http://arxiv.org/abs/1804.03599
[same with here]: https://github.com/1Konny/FactorVAE
[Github Repo]: https://github.com/miyosuda/disentangled_vae
[miyosuda]: https://github.com/miyosuda
