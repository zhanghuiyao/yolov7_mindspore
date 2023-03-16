#!/bin/bash

###==========================================================================
### Usage: bash run_standalone_test_ascend.sh [OPTIONS]...
### Description:
###     Run distributed test for YOLOv5 model.
###     Note that long option should use '--option=value' format, short option should use '-o value'
### Options:
###   -d  --data                path to dataset config yaml file
###   -D, --device              device id for standalone train, or start device id for distribute train
###   -H, --help                print this help message
###   -h  --hyp                 path to hyper-parameter config yaml file
###   -c, --config              path to model config file
###   -w, --weights              path to checkpoint weights
###   -w, --weights             ckpt to model
###   -m, --weights             mindir to model
### Example:
### 1. Test checkpoint with config. Configs in [] are optional.
###     bash run_standalone_test_ascend.sh -w weights.ckpt [-c config.yaml -d coco.yaml --hyp=hyp.config.yaml]
###==========================================================================


source common.sh
parse_args "$@"
get_default_config

export DEVICE_ID=$DEVICE_ID
export RANK_ID=0

echo "WEIGHTS: $WEIGHTS"
echo "CONFIG PATH: $CONFIG_PATH"
echo "DATA PATH: $DATA_PATH"
echo "HYP PATH: $HYP_PATH"
echo "DEVICE ID: $DEVICE_ID"

if [ -z "$WEIGHTS" ]; then
    echo "ERROR: Weights argument path is empty, which is required."
    exit 1
fi

cur_dir=$(pwd)
build_third_party_files "$cur_dir" "../third_party"

export DEVICE_NUM=1
export RANK_SIZE=1
eval_exp=$(get_work_dir "eval_exp_standalone")
eval_exp=$(realpath "${eval_exp}")
echo "Make directory ${eval_exp}"
copy_files_to "$eval_exp"
cd "${eval_exp}" || exit
env > env.log
python test.py \
  --weights=$WEIGHTS \
  --cfg=$CONFIG_PATH \
  --data=$DATA_PATH \
  --hyp=$HYP_PATH \
  --device_target=Ascend \
  --img_size=640 \
  --conf=0.001 \
  --rect=False \
  --iou_thres=0.65 \
  --batch_size=16 > log.txt 2>&1 &
cd ..
