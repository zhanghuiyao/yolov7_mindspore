#!/bin/bash

if [ $# != 2 ] && [ $# != 1 ]
then
    echo "Usage: bash run_distribute_train.sh [CONFIG_PATH] [DEVICE_ID]"
    echo "OR"
    echo "Usage: bash run_distribute_train.sh [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# == 1 ]
then
  DEVICE_ID=$1
  CONFIG_PATH=$"./config/yolov7/net/yolov7.yaml"
fi

if [ $# == 2 ]
then
  DEVICE_ID=$2
  CONFIG_PATH=$(get_real_path $1)
fi

echo $CONFIG_PATH
export DEVICE_ID=$DEVICE_ID
export RANK_ID=$DEVICE_ID
export DEVICE_NUM=1
export RANK_SIZE=1

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $RANK_SIZE`
gap=`expr $avg \- 1`

rm -rf ./train_standalone$DEVICE_ID
mkdir ./train_standalone$DEVICE_ID
cp ../*.py ./train_standalone$DEVICE_ID
cp -r ../config ./train_standalone$DEVICE_ID
cp -r ../mindyolo ./train_standalone$DEVICE_ID
mkdir ./train_standalone$DEVICE_ID/scripts
cp -r ../scripts/*.sh ./train_standalone$DEVICE_ID/scripts/
cd ./train_standalone$DEVICE_ID || exit
echo "start training for rank $RANK_ID, device $DEVICE_ID"
env > env.log
python train.py \
    --device_target=Ascend \
    --sync_bn=False \
    --config=$CONFIG_PATH \
    --is_distributed=False \
    --batch_size=16 > log.txt 2>&1 &
cd ..
