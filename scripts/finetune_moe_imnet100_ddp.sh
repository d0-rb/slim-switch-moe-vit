#!/bin/bash
read -p 'cuda: ' cuda
read -p 'gate: ' gate
read -p 'num-experts: ' num_experts
read -p 'lr: ' lr
read -p 'top-k :' topk
#read -p 'epochs: ' epochs

num_comma=`echo ${cuda} | tr -cd , | wc -c`
num_cuda=$((${num_comma} + 1))
echo $num_cuda
echo $num_comma

port=$((9000 + RANDOM % 1000))
model="moe_tiny_patch16_224"
start_threshold="0.5"
dataset="IMNET100"
n=0  # seed
validation_size=0.1
epochs=1200
batchsize=1024
datapath="ImageNet100"

#NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$cuda torchrun --nproc_per_node=$num_cuda --master_port=$port main.py --model $model --data-set $dataset \
CUDA_VISIBLE_DEVICES=$cuda python finetune.py --model $model --data-set $dataset\
		--unscale-lr \
        --data-path $datapath --batch $batchsize \
        --lr $lr --epochs $epochs --weight-decay 0.05 --sched cosine --input-size 224 \
        --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.1 --warmup-epochs 5 --drop 0.0 \
        --seed $n --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.0 --cutmix 1.0 \
        --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
		--num-experts $num_experts \
        --gate $gate \
        --validation-size $validation_size \
        --top-k $topk \
        --resume \
		pretrained/${dataset}/${model}/${gate}/lr_1e-3_ep_${epochs}/experts_${num_experts}/${n}/best_checkpoint.pth\
        --output_dir test
        #--output_dir finetune_v2/${dataset}/${model}/${gate}/lr_${lr}_ep_10_topk_${topk}/experts_${num_experts}/${n}/\
 #\

