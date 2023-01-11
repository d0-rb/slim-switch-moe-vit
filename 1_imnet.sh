#!/bin/bash

model=resmoe_tiny_patch16_224_expert8_attn_loss
CUDA=2
SEED=0
START_THRESHOLD_DENSE=0.9
TARGET_THRESHOLD_DENSE=0.5
START_THRESHOLD_MOE=0.9
TARGET_THRESHOLD_MOE=0.5
k=3
batch=$((256 * k))
LR=$((2 * k))e-4
EPOCH=300
dataset=IMNET

CUDA_VISIBLE_DEVICES=${CUDA} python main.py --model $model --data-set $dataset --data-path ./imagenet/ --batch $batch \
                --lr ${LR} --epochs ${EPOCH} --weight-decay 0.05 --sched cosine --input-size 224 \
                --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.1 --drop 0.0 \
                --seed ${SEED} --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.0 --cutmix 1.0 \
                --unscale-lr --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
                --starting-threshold-dense ${START_THRESHOLD_DENSE} --target-threshold-dense ${TARGET_THRESHOLD_DENSE} \
                --starting-threshold-moe ${START_THRESHOLD_MOE} --target-threshold-moe ${TARGET_THRESHOLD_MOE} \
                --warmup-epochs 5 --gate-lr ${LR} \
                --output_dir logs/$dataset/$model/scratch/lr_${LR}_ep_${EPOCH}/dense_${START_THRESHOLD_DENSE}_${TARGET_THRESHOLD_DENSE}/moe_${START_THRESHOLD_MOE}_${TARGET_THRESHOLD_MOE}/${SEED} \
		--resume logs/IMNET/resmoe_tiny_patch16_224_expert8_attn_loss/scratch/lr_6e-4_ep_300/dense_0.9_0.5/moe_0.9_0.5/0/best_checkpoint.pth
		#--output_dir test
