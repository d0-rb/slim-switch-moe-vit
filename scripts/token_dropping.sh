#!/bin/bash
read -p 'cuda: ' cuda
read -p 'num-experts: ' num_experts
read -p 'gate: ' gate # either naive or gshard
read -p 'num_experts: ' num_experts # 8/16/32


#####
# modify this param to connect to the dataset
dataset="IMNET100" # IMNET or IMNET100
datapath="./ImageNet100" # path to either dataset

lr="1e-3"

outputdir="outputs"
port=$((9000 + RANDOM % 1000))
model="moe_base_patch16_224"
start_threshold="0.5"
n=0  # seed
validation_size=0.001
epochs=300
batchsize=1024
#datapath="./imagenet"

resume_path="pretrained/${dataset}/${model}/${gate}/lr_1e-3_ep_${epochs}/experts_${num_experts}/val_${validation_size}/${n}/best_checkpoint.pth"

if [ ! -f $resume_path ]; then
    resume_path="pretrained/${dataset}/${model}/${gate}/lr_1e-3_ep_${epochs}/experts_${num_experts}/${n}/best_checkpoint.pth"
fi

output_path="$outputdir/${dataset}/${model}/${gate}/experts_${num_experts}/${n}/drop_tokens/"


CUDA_VISIBLE_DEVICES=$cuda python finetune.py --model $model --data-set $dataset\
    --unscale-lr \
    --data-path $datapath --batch $batchsize \
    --epochs $epochs --weight-decay 0.05 --sched cosine --input-size 224 \
    --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.1 --warmup-epochs 5 --drop 0.0 \
    --seed $n --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.0 --cutmix 1.0 \
    --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
	--num-experts $num_experts \
    --gate $gate \
    --validation-size $validation_size \
    --local-drop \
    --resume $resume_path\
    --output_dir $output_path\
