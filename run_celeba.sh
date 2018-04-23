#! /bin/sh

python main.py --dataset celeba --max_iter 1e6 --beta 64 --batch_size 64 --lr 1e-4 --z_dim 10 --viz_name beta_vae_celeba --viz_port 55558
