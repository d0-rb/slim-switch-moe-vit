#!/bin/bash
read -p 'model: ' model #="deit_tiny_patch16_224"
read -p 'cuda: ' cuda
read -p 'num_cuda: ' num_cuda
read -p 'seed: ' n #=0
read -p 'lr: ' lr #=0
# read -p 'starting threshold: ' start_threshold
# read -p 'target threshold: ' target_threshold
start_threshold="0.8"
target_threshold="0.8"

CUDA_VISIBLE_DEVICES=$cuda python3 -m torch.distributed.launch --nproc_per_node=$num_cuda --master_port=2322 --use_env main.py --model $model --data-set CIFAR10 \
                --data-path ./dataset --batch 256 \
                --lr $lr --epochs 400 --weight-decay 0.05 --sched cosine --input-size 224 \
                --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.1 --warmup-epochs 5 --drop 0.0 \
                --seed $n --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.0 --cutmix 1.0 \
                --unscale-lr --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
                --starting-threshold $start_threshold --target-threshold $target_threshold \
                --output_dir cifar-models/$model/scratch_${start_threshold}_${target_threshold}_${lr}/$n \
               # --finetune https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth
