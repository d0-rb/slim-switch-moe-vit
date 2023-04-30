#!/bin/bash
read -p 'cuda: ' cuda
#read -p 'gate: ' gate
#read -p 'num-experts: ' num_experts
read -p 'lr: ' lr
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
edt="cosinesim" # expert-drop-type
port=$((9000 + RANDOM % 1000))
model="moe_tiny_patch16_224"
start_threshold="0.5"
dataset="IMNET"
n=0  # seed
validation_size=0.001
epochs=300
batchsize=1024
# datapath="./imagenet"
datapath="../ImageNet"

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
        --resume \
	pretrained/${dataset}/${model}/${gate}/lr_1e-3_ep_${epochs}/experts_${num_experts}/val_${validation_size}/${n}/best_checkpoint.pth\
	--expert-keep-rate $edr --expert-drop-type $edt --expert-drop-local \
	--top-k $topk \
    --output_dir finetune_imnet/${dataset}/${model}/${gate}/lr_${lr}_ep_10_hubme/experts_${num_experts}/${n}/\
