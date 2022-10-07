#!/bin/bash


if [ $# != 3 ]
then
    echo "Usage: sh run_distribute_train_gpu.sh [DATA_CONFIG_PATH] [MODEL_CONFIG_PATH] [HYP_CONFIG_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_CONFIG_PATH=$(get_real_path $1)
MODEL_CONFIG_PATH=$(get_real_path $2)
HYP_CONFIG_PATH=$(get_real_path $2)
echo DATA_CONFIG_PATH
echo MODEL_CONFIG_PATH
echo HYP_CONFIG_PATH

if [ ! -f $DATA_CONFIG_PATH ]
then
    echo "error: DATA_CONFIG_PATH=$DATA_CONFIG_PATH is not a file"
exit 1
fi

if [ ! -f $MODEL_CONFIG_PATH ]
then
    echo "error: MODEL_CONFIG_PATH=$MODEL_CONFIG_PATH is not a file"
exit 1
fi

if [ ! -f $HYP_CONFIG_PATH ]
then
    echo "error: HYP_CONFIG_PATH=$HYP_CONFIG_PATH is not a file"
exit 1
fi


export DEVICE_NUM=8
rm -rf ./train_parallel
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp -r ../config ./train_parallel
cp -r ../network ./train_parallel
cp -r ../utils ./train_parallel
cd ./train_parallel || exit
env > env.log
mpirun --allow-run-as-root -n ${DEVICE_NUM} --output-filename log_output --merge-stderr-to-stdout \
python train.py \
  --is_distributed True \
  --device_target GPU \
  --rank_size 8 \
  --rank 0 \
  --batch-size 32 \
  --data ./config/data/coco.yaml \
  --img 640 640 \
  --cfg ./config/network_yolov7/yolov7.yaml \
  --weights '' \
  --name yolov7 \
  --hyp ./config/data/hyp.scratch.p5.yaml > log.txt 2>&1 &
cd ..
