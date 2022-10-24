#!/bin/bash


if [ $# != 3 ] && [ $# != 0 ]
then
    echo "Usage: sh run_distribute_train.sh [CONFIG_PATH] [DATA_PATH] [HYP_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# == 0 ]
then
  CONFIG_PATH=$"./config/network_yolov7/yolov7.yaml"
  DATA_PATH=$"./config/data/coco.yaml"
  HYP_PATH=$"./config/data/hyp.scratch.p5.yaml"
fi

if [ $# == 3 ]
then
  CONFIG_PATH=$(get_real_path $1)
  DATA_PATH=$(get_real_path $2)
  HYP_PATH=$(get_real_path $3)
fi

echo $CONFIG_PATH
echo $DATA_PATH
echo $HYP_PATH


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
  --ms_strategy="StaticShape" \
  --ms_loss_scaler="static" \
  --cfg=$CONFIG_PATH \
  --data=$DATA_PATH \
  --hyp=$HYP_PATH \
  --device_target=GPU \
  --is_distributed=True \
  --epochs=300 \
  --batch-size=96 > log.txt 2>&1 &
cd ..