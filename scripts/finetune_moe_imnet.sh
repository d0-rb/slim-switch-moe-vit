#!/bin/bash
read -p 'cuda: ' cuda
# read -p 'data path: ' datapath
read -p 'batch size: ' batchsize
# read -p 'gate: ' gate
read -p 'num-experts: ' num_experts
# read -p 'epochs: ' epochs
read -p 'n: ' n

num_comma=`echo ${cuda} | tr -cd , | wc -c`
num_cuda=$((${num_comma} + 1))

port=$((9000 + RANDOM % 1000))
model="moe_tiny_patch16_224"
lr="4e-5"
dataset="IMNET"
# n=0  # seed
datapath="../ImageNet"
# gate="naive"
# num_experts="32"
# batchsize="256"
# epochs="50"
epochs="0"
pretrained_epochs="300"
pretrained_lr="1e-3"
validation_size=0.001 # rougly 1k valid sample

# for gate in "gshard" "naive";
for gate in "gshard";
do
    # for keeprate in "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9";
    for keepcount in {0..48..8};
    do
        for droptype in "cosinesim" "meanshift" "volume";
        # for droptype in "random";
        do
            for droplocal in "false" "true";
            # for droplocal in "false";
            do
                for softmax_rescale in "false" "true";
                do
                    # CUDA_VISIBLE_DEVICES=$cuda python finetune.py --model $model --data-set $dataset \
                    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$cuda torchrun --nproc_per_node=$num_cuda --master_port=$port finetune.py --model $model --data-set $dataset \
                            --unscale-lr \
                            --data-path $datapath --batch $batchsize \
                            --lr $lr --epochs $epochs --weight-decay 0.05 --sched cosine --input-size 224 \
                            --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.1 --warmup-epochs 5 --drop 0.0 \
                            --seed $n --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.0 --cutmix 1.0 \
                            --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
                            --num-experts $num_experts \
                            --gate $gate \
                            --validation-size $validation_size \
                            --expert-keep-count $keepcount \
                            --expert-drop-type $droptype \
                            --expert-drop-local $droplocal \
                            --softmax-rescale $softmax_rescale \
                            --finetune \
                            pretrained/${dataset}/${model}/${gate}/lr_${pretrained_lr}_ep_${pretrained_epochs}/experts_${num_experts}/val_${validation_size}/0/best_checkpoint.pth \
                            --output_dir \
                            finetuned/${dataset}/${model}/${gate}/lr_${lr}_ep_${epochs}/experts_${num_experts}/${droptype}/droplocal_${droplocal}/softmax_rescale_${softmax_rescale}/${n}
                            # pretrained/${dataset}/${model}/${gate}/lr_${pretrained_lr}_ep_${pretrained_epochs}/experts_${num_experts}/val_${validation_size}/0/best_checkpoint.pth \
                            # --expert-keep-rate $keeprate \
                done
            done
        done
    done
done
