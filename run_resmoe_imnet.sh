#!/bin/bash
read -p 'cuda: ' cuda
read -p 'data path: ' datapath

num_comma=`echo ${cuda} | tr -cd , | wc -c`
num_cuda=$((${num_comma} + 1))

port=$((9000 + RANDOM % 1000))
model="resmoe_tiny_patch16_224_expert8_attn_loss_v2"
lr="2e-4"
start_threshold="0.9"
target_threshold="0.5"
dataset="IMNET"
n=0  # seed
epochs=400

CUDA_VISIBLE_DEVICES=$cuda python3 -m torch.distributed.launch --nproc_per_node=$num_cuda --master_port=$port --use_env main.py --model $model --data-set $dataset \
                --data-path $datapath --batch 128 \
                --lr $lr --epochs $epochs --weight-decay 0.05 --sched cosine --input-size 224 \
                --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.1 --warmup-epochs 5 --drop 0.0 \
                --seed $n --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.0 --cutmix 1.0 \
                --unscale-lr --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
                --starting-threshold-dense $start_threshold --target-threshold-dense $target_threshold \
                --starting-threshold-moe $start_threshold --target-threshold-moe $target_threshold \
                --output_dir imnet-models/$model/scratch_${start_threshold}_${target_threshold}_${lr}/$n \
                # --num-tasks $num_tasks \
                # --rehearsal --rehearsal-batch-size 512 \
               # --finetune https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth
