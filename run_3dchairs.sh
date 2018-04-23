#! /bin/sh

python main.py --dataset 3dchairs --max_iter 1e6 --beta 4 --batch_size 64 --lr 1e-4 --z_dim 10 --viz_name beta_vae_3dchairs --viz_port 55558
