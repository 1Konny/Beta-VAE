#! /bin/sh

python main.py --dataset celeba --seed 1 --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 64 --z_dim 10 --max_iter 1.5e6 \
    --beta 10 --viz_name celeba_H_beta10_z10
