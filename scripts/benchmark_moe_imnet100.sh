#!/bin/bash
read -p 'cuda: ' cuda
# read -p 'data path: ' datapath
read -p 'batch size: ' batchsize
# read -p 'gate: ' gate
# read -p 'num-experts: ' num_experts
# read -p 'n: ' n

num_comma=`echo ${cuda} | tr -cd , | wc -c`
num_cuda=$((${num_comma} + 1))

port=$((9000 + RANDOM % 1000))
model="moe_tiny_patch16_224"
lr="4e-5"
dataset="IMNET100"
n=0  # seed
# gate="naive"
num_experts="32"
# batchsize="256"
keeprates="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
epochs="50"

for gate in "gshard" "naive";
do
    for droptype in "random" "cosinesim" "norm" "meanshift" "volume";
    do
        CUDA_VISIBLE_DEVICES=$cuda \
        python benchmark.py --batch_size 256 --model $model \
                --input_size 3 224 224 \
                --seed $n \
                --num_experts $num_experts \
                --gate $gate \
                --num_classes 100 \
                --keeprates ${keeprates} \
                --resume \
                finetuned/${dataset}/${model}/${gate}/lr_${lr}_ep_${epochs}/experts_${num_experts}/${droptype} \
                --output_dir \
                finetuned/${dataset}/${model}/${gate}/lr_${lr}_ep_${epochs}/experts_${num_experts}/${droptype}/benchmarks
    done
done
