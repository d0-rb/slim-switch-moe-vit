#!/bin/bash
read -p 'model: ' model #="deit_tiny_patch16_224"
read -p 'cuda: ' cuda
read -p 'seed: ' n #=0
read -p 'starting threshold: ' start_threshold
read -p 'target threshold: ' target_threshold
read -p 'lr: ' lr

epoch=400
CUDA_VISIBLE_DEVICES=$cuda python main.py --model $model --data-set CIFAR10 --data-path ./dataset --batch 156 \
                --lr $lr --epochs $epoch --weight-decay 0.05 --sched cosine --input-size 224 \
                --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.1 --drop 0.0 \
                --seed $n --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.0 --cutmix 1.0 \
                --unscale-lr --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
                --starting-threshold-dense ${start_threshold} --target-threshold-dense ${target_threshold} \
                --starting-threshold-moe ${start_threshold} --target-threshold-moe ${target_threshold} \
                --output_dir logs/cifar10/$model/scratch/lr_${lr}_ep_${epoch}/thres_${start_threshold}_${target_threshold}/${n} \
		--gate-lr $lr
                # --vis-enabled
                # --resume logs/cifar10/$model/scratch/lr_${lr}_ep_${epoch}/thres_${start_threshold}_${target_threshold}/${n}/checkpoint.pth \
               # --finetune https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth
