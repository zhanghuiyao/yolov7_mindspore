#!/bin/bash

###==========================================================================
### Usage: bash run_standalone_train_ascend.sh [OPTIONS]...
### Description:
###     Run distributed train for YOLOv5 model.
###     Note that long option should use '--option=value' format, short option should use '-o value'
### Options:
###   -d  --data                path to dataset config yaml file
###   -D, --device              device id for standalone train, or start device id for distribute train
###   -H, --help                print this help message
###   -h  --hyp                 path to hyper-parameter config yaml file
###   -c, --config              path to model config file
###   -w, --weights             ckpt to model
###   -m, --weights             mindir to model
### Example:
### 1. train models with config. Configs in [] are optional.
###     bash run_standalone_train_ascend.sh [-c config.yaml -d coco.yaml --hyp=hyp.config.yaml]
###==========================================================================

source common.sh
parse_args "$@"
get_default_config

export DEVICE_ID=$DEVICE_ID
echo "CONFIG PATH: $CONFIG_PATH"
echo "DATA PATH: $DATA_PATH"
echo "HYP PATH: $HYP_PATH"
echo "DEVICE ID: $DEVICE_ID"

cur_dir=$(pwd)
build_third_party_files "$cur_dir" "../third_party"

export RANK_ID=0
train_exp=$(get_work_dir "train_exp_standalone")
train_exp=$(realpath "${train_exp}")
echo "Make directory ${train_exp}"
copy_files_to "$train_exp"
cd "${train_exp}" || exit
env > env.log

python train.py \
        --clip_grad=False \
        --optimizer="momentum" \
        --cfg=$CONFIG_PATH \
        --data=$DATA_PATH \
        --hyp=$HYP_PATH \
        --device_target=Ascend \
        --profiler=False \
        --accumulate=False \
        --epochs=300 \
        --iou_thres=0.65 \
        --batch_size=16  > log.txt 2>&1 &
cd ..

