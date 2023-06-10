#!/bin/bash
read -p 'cuda: ' cuda
#read -p 'gate: ' gate
#read -p 'num-experts: ' num_experts
read -p 'lr: ' lr
read -p 'edt: ' edt
#read -p 'top-k :' topk
#read -p 'epochs: ' epochs
num_experts=32
gate="gshard"
lr="1e-3"
num_comma=`echo ${cuda} | tr -cd , | wc -c`
num_cuda=$((${num_comma} + 1))
echo $num_cuda
echo $num_comma

topk=1
edr=0.5 # expert-drop-ratio
#edt="cosines,m" # expert-drop-type
port=$((9000 + RANDOM % 1000))
model="moe_tiny_patch16_224"
start_threshold="0.5"
dataset="IMNET"
n=0  # seed
validation_size=0.001
epochs=300
batchsize=1024
datapath="./imagenet"
#datapath="../ImageNet"

#pretrained_gshard/${dataset}/${model}/${gate}/lr_1e-3_ep_${epochs}/experts_${num_experts}/val_${validation_size}/${n}/best_checkpoint.pth\
#NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$cuda torchrun --nproc_per_node=$num_cuda --master_port=$port main.py --model $model --data-set $dataset \
CUDA_VISIBLE_DEVICES=$cuda python generator.py --model $model --data-set $dataset\
    --unscale-lr \
    --data-path $datapath --batch $batchsize\
    --seed $n\
    --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
	--num-experts $num_experts \
    --gate $gate \
    --validation-size $validation_size \
    --resume \
        pretrained/${dataset}/${model}/${gate}/lr_1e-3_ep_${epochs}/experts_${num_experts}/${n}/best_checkpoint.pth\
    --expert-keep-rate $edr --expert-drop-type $edt --expert-drop-local false \
    --drop_tokens \
    --top-k $topk \
    --output_dir finetune_imnet/${dataset}/${model}/${gate}/lr_${lr}_ep_10_hubme/experts_${num_experts}/${n}/\
