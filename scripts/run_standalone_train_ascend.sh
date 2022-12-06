#!/bin/bash

if [ $# != 4 ] && [ $# != 1 ]
then
    echo "Usage: bash run_distribute_train.sh [CONFIG_PATH] [DATA_PATH] [HYP_PATH] [DEVICE_ID]"
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
  CONFIG_PATH=$"./config/network_yolov7/yolov7.yaml"
  DATA_PATH=$"./config/data/coco.yaml"
  HYP_PATH=$"./config/data/hyp.scratch.p5.yaml"
fi

if [ $# == 4 ]
then
  DEVICE_ID=$4
  CONFIG_PATH=$(get_real_path $1)
  DATA_PATH=$(get_real_path $2)
  HYP_PATH=$(get_real_path $3)
fi

echo $CONFIG_PATH
echo $DATA_PATH
echo $HYP_PATH


export DEVICE_NUM=1
export RANK_SIZE=1
export DEVICE_ID=$DEVICE_ID
export RANK_ID=$DEVICE_ID

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $RANK_SIZE`
gap=`expr $avg \- 1`

rm -rf ./train_standalone$DEVICE_ID
mkdir ./train_standalone$DEVICE_ID
cp ../*.py ./train_standalone$DEVICE_ID
cp -r ../config ./train_standalone$DEVICE_ID
cp -r ../network ./train_standalone$DEVICE_ID
cp -r ../utils ./train_standalone$DEVICE_ID
mkdir ./train_standalone$DEVICE_ID/scripts
cp -r ../scripts/*.sh ./train_standalone$DEVICE_ID/scripts/
cd ./train_standalone$DEVICE_ID || exit
echo "start training for rank $RANK_ID, device $DEVICE_ID"
env > env.log
python train.py \
    --ms_strategy="StaticShape" \
    --ms_amp_level="O0" \
    --ms_loss_scaler="static" \
    --ms_loss_scaler_value=1024 \
    --ms_optim_loss_scale=1 \
    --ms_grad_sens=1024 \
    --overflow_still_update=True \
    --clip_grad=False \
    --sync_bn=False \
    --optimizer="momentum" \
    --cfg=$CONFIG_PATH \
    --data=$DATA_PATH \
    --hyp=$HYP_PATH \
    --device_target=Ascend \
    --is_distributed=False \
    --epochs=300 \
    --recompute=True \
    --recompute_layers=5 \
    --batch_size=16  > log.txt 2>&1 &
cd ..
