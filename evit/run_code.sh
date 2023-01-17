now=$(date +"%Y%m%d_%H%M%S")
# logdir=/train_log/exp_$now
logdir=../benchmarks/tiny/moe-evit/0
datapath="/ssd1/xinyu/dataset/imagenet2012"

echo "output dir: $logdir"

# python3 -m torch.distributed.launch --nproc_per_node=3 --master_port=$RANDOM --use_env \
python3 \
	main.py \
	--model moe_deit_tiny_patch16_224_shrink_base_expert8 \
	--fuse_token \
	--base_keep_rate 0.7 \
	--input-size 224 \
	--batch-size 512 \
	--warmup-epochs 5 \
	--shrink_start_epoch 10 \
	--shrink_epochs 100 \
	--epochs 300 \
	--dist-eval \
	--data-path $datapath \
	--output_dir $logdir \
    --resume ../benchmarks/tiny/moe-evit/0/checkpoint.pth

echo "output dir for the last exp: $logdir"\
