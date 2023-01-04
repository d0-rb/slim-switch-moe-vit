#!/bin/bash
read -p 'model: ' model #="deit_tiny_patch16_224"
read -p 'cuda: ' cuda
read -p 'num_cuda: ' num_cuda
CUDA_VISIBLE_DEVICES=$cuda
# read -p 'starting threshold: ' starting_threshold
starting_threshold=1.0
# read -p 'target threshold: ' target_threshold
target_threshold=1.0

for n in 0 1 2
do
    for start_threshold in 1.0 0.9 0.7 0.5
    do
        for target_threshold_diff in 0.0 0.1 0.2 0.4
        do
            target_threshold=$(bc <<< "$start_threshold - $target_threshold_diff")

            python3 -m torch.distributed.launch --nproc_per_node=$num_cuda --master_port=$RANDOM --use_env main.py --model $model --data-set CIFAR10 --data-path ./dataset --batch 128\
                --lr 1e-3 --epochs 25 --weight-decay 0.05 --sched cosine --input-size 224\
                --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.1 --warmup-epochs 5 --drop 0.0 \
                --seed $n --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.0 --cutmix 1.0 \
                --unscale-lr --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
                --starting-threshold $start_threshold --target-threshold $target_threshold \
                --output_dir cifar-models/$model/scratch_${start_threshold}_${target_threshold}/$n\
               # --finetune https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth
        done
    done
done
