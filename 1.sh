#!/bin/bash

model=moe_tiny_patch16_224_expert8
CUDA=0
SEED=0
START_THRESHOLD=0.9
TARGET_THRESHOLD=0.5
LR=2e-4
EPOCH=400

CUDA_VISIBLE_DEVICES=${CUDA} python main.py --model $model --data-set CIFAR10 --data-path ./dataset --batch 156 \
                --lr ${LR} --epochs ${EPOCH} --weight-decay 0.05 --sched cosine --input-size 224 \
                --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.1 --drop 0.0 \
                --seed ${SEED} --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.0 --cutmix 1.0 \
                --unscale-lr --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
                --starting-threshold ${START_THRESHOLD} --target-threshold ${TARGET_THRESHOLD} \
                --output_dir logs/cifar10/$model/scratch/lr_${LR}_ep_${EPOCH}/thres_${START_THRESHOLD}_${TARGET_THRESHOLD}/${SEED} \
                --warmup-epochs 5 --gate-lr ${LR}
