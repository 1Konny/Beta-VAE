#! /bin/sh

python main.py --dataset dsprites --max_iter 3e5 --beta 4 --batch_size 64 --lr 1e-4 --z_dim 10 --viz_name beta_vae_dsprites --viz_port 55558
