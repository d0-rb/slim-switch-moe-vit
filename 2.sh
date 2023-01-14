#!/bin/bash

MODEL=resmoe_tiny_patch16_224_expert8_attn_loss_v4
CUDA=1
SEED=0
START_THRESHOLD_DENSE=0.9
TARGET_THRESHOLD_DENSE=0.5
START_THRESHOLD_MOE=0.9
TARGET_THRESHOLD_MOE=0.5
LR=3e-4
EPOCH=300

CUDA_VISIBLE_DEVICES=${CUDA} python main.py --model ${MODEL} --data-set CIFAR10 --data-path ./dataset --batch 156 \
                --lr ${LR} --epochs ${EPOCH} --weight-decay 0.05 --sched cosine --input-size 224 \
                --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.1 --drop 0.0 \
                --seed ${SEED} --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.0 --cutmix 1.0 \
                --unscale-lr --no-repeated-aug --aa rand-m9-mstd0.5-inc1 --threshold-scheduler linear \
                --starting-threshold-dense ${START_THRESHOLD_DENSE} --target-threshold-dense ${TARGET_THRESHOLD_DENSE} \
                --starting-threshold-moe ${START_THRESHOLD_MOE} --target-threshold-moe ${TARGET_THRESHOLD_MOE} \
                --warmup-epochs 5 --gate-lr ${LR} \
                --output_dir logs/cifar10/${MODEL}/scratch/lr_${LR}_ep_${EPOCH}/linear_thres/dense_${START_THRESHOLD_DENSE}_${TARGET_THRESHOLD_DENSE}/moe_${START_THRESHOLD_MOE}_${TARGET_THRESHOLD_MOE}/${SEED}
