#!/bin/bash

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: bash run_standalone_test_ascend.sh [WEIGHTS] [DEVICE_ID]"
    echo "OR"
    echo "Usage: bash run_standalone_test_ascend.sh [CONFIG_PATH] [WEIGHTS] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# == 2 ]
then
  WEIGHTS=$1
  DEVICE_ID=$2
  CONFIG_PATH=$"./config/yolov7/net/yolov7.yaml"
fi

if [ $# == 3 ]
then
  WEIGHTS=$2
  DEVICE_ID=$3
  CONFIG_PATH=$(get_real_path $1)
fi

echo $CONFIG_PATH
export RANK_ID=$DEVICE_ID
export DEVICE_ID=$DEVICE_ID
export DEVICE_NUM=1
export RANK_SIZE=1
rm -rf ./test_standalone$DEVICE_ID
mkdir ./test_standalone$DEVICE_ID
cp ../*.py ./test_standalone$DEVICE_ID
cp -r ../config ./test_standalone$DEVICE_ID
cp -r ../mindyolo ./test_standalone$DEVICE_ID
mkdir ./test_standalone$DEVICE_ID/scripts
cp -r ../scripts/*.sh ./test_standalone$DEVICE_ID/scripts/
cd ./test_standalone$DEVICE_ID || exit
env > env.log
python test.py \
  --weights=$WEIGHTS \
  --config=$CONFIG_PATH \
  --device_target=Ascend \
  --img_size=640 \
  --conf=0.001 \
  --iou=0.65 \
  --batch_size=32 > log.txt 2>&1 &
cd ..