#!/bin/bash
read -p 'cuda: ' cuda
read -p 'data path: ' datapath
read -p 'batch size: ' batchsize
read -p 'target_threshold_edge ' target_threshold_edge
read -p 'target_threshold_node ' target_threshold_node
read -p 'num rep ' num_rep

num_comma=`echo ${cuda} | tr -cd , | wc -c`
num_cuda=$((${num_comma} + 1))
echo $num_cuda
echo $num_comma

port=$((9000 + RANDOM % 1000))
#model="resmoe_tiny_patch16_224_expert8_attn_loss_v4_delay_start"
#model="deit_tiny_patch16_224"
model='resvit_tiny_patch16_224_gcn_g3_o14'
lr="3e-4"
start_threshold="0.5"
dataset="IMNET100"
n=0  # seed
epochs=300

CUDA_VISIBLE_DEVICES=$cuda python  main.py --model $model --data-set $dataset \
                --data-path $datapath --batch $batchsize \
                --lr $lr --epochs $epochs --weight-decay 0.05 --sched cosine --input-size 224 \
                --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.1 --warmup-epochs 5 --drop 0.0 \
                --seed $n --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.0 --cutmix 1.0 \
                --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
		--num-rep $num_rep \
                --starting-threshold-dense $start_threshold --target-threshold-dense $target_threshold_node \
                --starting-threshold-moe $start_threshold --target-threshold-moe $target_threshold_edge \
                --output_dir "test/"\
                --debug
