#!/bin/bash


if [ $# != 1 ] && [ $# != 0 ]
then
    echo "Usage: bash run_distribute_train_gpu.sh [CONFIG_PATH]"
    echo "OR"
    echo "Usage: bash run_distribute_train_gpu.sh"
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
  CONFIG_PATH=$"./config/yolov7/net/yolov7.yaml"
fi

if [ $# == 1 ]
then
  CONFIG_PATH=$(get_real_path $1)
fi

echo $CONFIG_PATH
export DEVICE_NUM=8
rm -rf ./train_parallel
mkdir ./train_parallel
cp ../*.py ./train_parallel
cp -r ../config ./train_parallel
cp -r ../mindyolo ./train_parallel
mkdir ./train_parallel/scripts
cp -r ../scripts/*.sh ./train_parallel/scripts/
cd ./train_parallel || exit
env > env.log
mpirun --allow-run-as-root -n ${DEVICE_NUM} --output-filename log_output --merge-stderr-to-stdout \
python train.py \
  --device_target=GPU \
  --overflow_still_update=False \
  --config=$CONFIG_PATH \
  --is_distributed=True > log.txt 2>&1 &
cd ..