#!/bin/bash

if [ $# != 2 ] && [ $# != 5 ]
then
    echo "Usage: sh run_standalone_eval_gpu.sh [WEIGHTS] [DEVICE_ID]"
    echo "OR"
    echo "Usage: sh run_standalone_eval_gpu.sh [WEIGHTS] [DEVICE_ID] [CONFIG_PATH] [DATA_PATH] [HYP_PATH]"
exit 1
fi

WEIGHTS=$1
export CUDA_VISIBLE_DEVICES=$2

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# == 2 ]
then
  CONFIG_PATH=$"./config/network_yolov7/yolov7.yaml"
  DATA_PATH=$"./config/data/coco.yaml"
  HYP_PATH=$"./config/data/hyp.scratch.p5.yaml"
fi

if [ $# == 5 ]
then
  CONFIG_PATH=$(get_real_path $3)
  DATA_PATH=$(get_real_path $4)
  HYP_PATH=$(get_real_path $5)
fi

echo $CONFIG_PATH
echo $DATA_PATH
echo $HYP_PATH


export DEVICE_NUM=1
rm -rf ./test_standalone$2
mkdir ./test_standalone$2
cp ../*.py ./test_standalone$2
cp -r ../config ./test_standalone$2
cp -r ../network ./test_standalone$2
cp -r ../utils ./test_standalone$2
mkdir ./test_standalone$2/scripts
cp -r ../scripts/*.sh ./test_standalone$2/scripts/
cd ./test_standalone$2 || exit
env > env.log
python test.py \
  --weights=$WEIGHTS \
  --cfg=$CONFIG_PATH \
  --data=$DATA_PATH \
  --hyp=$HYP_PATH \
  --device_target=GPU \
  --img-size=640 \
  --conf=0.001 \
  --iou=0.65 \
  --batch-size=32 > log.txt 2>&1 &
cd ..