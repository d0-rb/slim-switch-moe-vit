#!/bin/bash
read -p 'model: ' model #="deit_tiny_patch16_224"
read -p 'cuda: ' cuda
n=1
CUDA_VISIBLE_DEVICES=$cuda python main.py --model $model --data-set CIFAR10 --data-path ./dataset --batch 128\
                --lr 1e-3 --epochs 300 --weight-decay 0.05 --sched cosine --input-size 224\
                --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.1 --warmup-epochs 5 --drop 0.0 \
                --seed $n --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.0 --cutmix 1.0 \
                --unscale-lr --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
                --output_dir cifar-models/$model/scratch/$n\
               # --finetune https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth
