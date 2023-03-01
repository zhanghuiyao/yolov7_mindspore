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
model=$MINDIR
echo "CONFIG PATH: $CONFIG_PATH"
echo "DATA PATH: $DATA_PATH"
echo "HYP PATH: $HYP_PATH"
echo "DEVICE ID: $DEVICE_ID"
echo "model: $MINDIR"

function compile_app()
{
    cd ascend310_infer || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log

    if [ $? -ne 0 ]; then
        echo "compile app code failed"
        exit 1
    fi
    cd - || exit
}

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python preprocess.py --output_path=./preprocess_Result
}

function infer()
{
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    ascend310_infer/out/main --model_path=$model --dataset_path=$data_path --device_id=$device_id &> infer.log

    if [ $? -ne 0 ]; then
        echo "execute inference failed"
        exit 1
    fi
}

function cal_acc()
{
    python postprocess.py --result_path=result_Files &> acc.log

    if [ $? -ne 0 ]; then
        echo "calculate accuracy failed"
        exit 1
    fi
}

preprocess_data
data_path=./preprocess_Result/img_data
compile_app
infer
cal_acc