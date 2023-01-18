#!/bin/bash

if [ $# != 2 ] && [ $# != 1 ]
then
    echo "Usage: bash run_distribute_train.sh [CONFIG_PATH] [RANK_TABLE_FILE]"
    echo "OR"
    echo "Usage: bash run_distribute_train.sh [RANK_TABLE_FILE]"
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
  RANK_TABLE_FILE=$(get_real_path $1)
  CONFIG_PATH=$"./config/yolov7/net/yolov7.yaml"
fi

if [ $# == 2 ]
then
  CONFIG_PATH=$(get_real_path $1)
  RANK_TABLE_FILE=$(get_real_path $2)
fi

echo $CONFIG_PATH
echo $RANK_TABLE_FILE

if [ ! -f $RANK_TABLE_FILE ]
then
    echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a file"
exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$RANK_TABLE_FILE
export MINDSPORE_HCCL_CONFIG_PATH=$RANK_TABLE_FILE

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $RANK_SIZE`
gap=`expr $avg \- 1`

for((i=0; i<${DEVICE_NUM}; i++))
do
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end

    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp -r ../config ./train_parallel$i
    cp -r ../mindyolo ./train_parallel$i
    mkdir ./train_parallel$i/scripts
    cp -r ../scripts/*.sh ./train_parallel$i/scripts/
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    taskset -c $cmdopt python train.py \
        --device_target=Ascend \
        --sync_bn=True \
        --config=$CONFIG_PATH \
        --is_distributed=True > log.txt 2>&1 &
    cd ..
done
