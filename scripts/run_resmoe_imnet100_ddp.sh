#!/bin/bash
cuda="4,5,6,7"
batchsize=256
model="resmoe_tiny_patch16_224_expert8_attn_loss_v4"
lr="3e-4"
start_threshold="0.9"
target_threshold="0.5"
datapath="~/DATASET/ImageNet100"
dataset="IMNET100"
n=0  # seed
epochs=300
num_cuda=4
port=$((9000 + RANDOM % 1000))

CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=$num_cuda --master_port=$port --use_env \
                main.py --model $model --data-set $dataset --data-path $datapath --batch $batchsize \
                --lr $lr --unscale-lr --epochs $epochs --weight-decay 0.05 --sched cosine --input-size 224 \
                --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.1 --warmup-epochs 5 --drop 0.0 \
                --seed $n --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.0 --cutmix 1.0 \
                --unscale-lr --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
                --starting-threshold-dense $start_threshold --target-threshold-dense $target_threshold \
                --starting-threshold-moe $start_threshold --target-threshold-moe $target_threshold \
                --output_dir logs/${dataset}/${model}/scratch/lr_${lr}_ep_${epochs}/cosine/dense_${start_threshold}_${target_threshold}/moe_${start_threshold}_${target_threshold}/${n}
