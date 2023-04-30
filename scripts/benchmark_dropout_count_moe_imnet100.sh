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
scan_keeprates="197"
droptype="random"
epochs="50"

for gate in "gshard" "naive";
do
    for droplocal in "false" "true";
    do
        CUDA_VISIBLE_DEVICES=$cuda \
        python benchmark.py --batch_size ${batchsize} --model $model \
                --input_size 3 224 224 \
                --seed $n \
                --num_experts $num_experts \
                --gate $gate \
                --num_classes 100 \
                --scan-keeprates ${scan_keeprates} \
                --resume \
                finetuned/${dataset}/${model}/${gate}/lr_${lr}_ep_${epochs}/experts_${num_experts} \
                --output_dir \
                finetuned/${dataset}/${model}/${gate}/lr_${lr}_ep_${epochs}/experts_${num_experts}/benchmarks/batchsize_${batchsize}/droplocal_${droplocal} \
                --expert-drop-local $droplocal
    done
done
