#!/bin/bash

# If $__COMMON_SH__ is not NULL, skip import script
if [ -n "$__COMMON_SH__" ]; then
  return
fi

__COMMON_SH__='common.sh'

function build_third_party_files() {
  cur_dir=$(realpath -e "$1")
  third_party_dir=$(realpath -e "$2")
  for file in "${third_party_dir}"/fast_coco/fast_coco_eval*.so; do
    if [ -e "$file" ]; then
      echo "fast_coco_eval library exists. Skip building."
    else
      cd "${third_party_dir}"/fast_coco || exit
      bash build.sh
      rm -rf build
      cd "${cur_dir}" || exit
    fi
  done
  for file in "${third_party_dir}"/fast_nms/fast_cpu_nms*.so; do
      if [ -e "$file" ]; then
        echo "fast_cpu_nms library exists. Skip building."
      else
        cd "${third_party_dir}"/fast_nms || exit
        bash build.sh
        rm -rf build
        cd "${cur_dir}" || exit
    fi
  done
}


function get_work_dir() {
    prefix=$1
    cnt=0
    dir_exp="${prefix}${cnt}"
    while [ -d "$dir_exp" ]
    do
      cnt=$((cnt + 1))
      dir_exp="${prefix}${cnt}"
    done
    echo "$dir_exp"
}

# ===================== SHELL SCRIPT ARGUMENTS REGULAR EXPRESSION =====================
SHORT_OPTION_RE="^-([a-zA-Z]+)"
LONG_OPTION_RE="^--([a-zA-Z]+[-a-zA-Z]*)=?(.*)"
NUMBER_RE="^[0-9]+$"
# =====================================================================================


# ===================== SHELL SCRIPT ARGUMENTS =====================
RANK_TABLE_FILE=""
CONFIG_PATH=""
DATA_PATH=""
HYP_PATH=""
WEIGHTS=""
MINDIR=""
DEVICE_ID=""
DEVICE_NUM=""
# ==================================================================

function usage() {
    sed -n 's/^###[ ]//;T;p' "$0"
}


function check_empty() {
    if [ -z "$2" ]; then
        echo "Argument for option $1 is empty."
        exit 1
    fi
}

function check_number() {
    if [[ ! "$2" =~ $NUMBER_RE ]]; then
        echo "Argument $2 for option $1 is not a valid number. You may type an invalid number or forget to type it."
        exit 1
    fi
}


function copy_files_to() {
  target_dir=$1
  mkdir "$target_dir"
  cp ../*.py "$target_dir"
  cp -r ../config "$target_dir"
  cp -r ../utils "$target_dir"
  cp -r ../network "$target_dir"
  if [ -d ../third_party ]; then
    cp -r ../third_party "$target_dir"
  fi
  mkdir "$target_dir/scripts"
  cp -r ./*.sh "$target_dir/scripts/"
}


function parse_args() {
    while [ "$#" -gt 0 ]; do
        if [[ $1 =~ $SHORT_OPTION_RE ]]; then
#            echo "Short option: ${BASH_REMATCH[1]}"
            option=${BASH_REMATCH[1]}
            value=$2
        elif [[ $1 =~ $LONG_OPTION_RE ]]; then
#            echo "Long option: ${BASH_REMATCH[1]}"
            option=${BASH_REMATCH[1]}
            value=${BASH_REMATCH[2]}
        else
            echo "Unknown option: $1. Please see usage:"; usage; exit 1;
        fi
        case $option in
            d | data)
                check_empty "$option" "$value"; DATA_PATH=$(realpath -e "$value");
                [ "$option" = "d" ] && shift;;
            H | help)
                usage; exit 0;;
            h | hyp)
                check_empty "$option" "$value"; HYP_PATH=$(realpath -e "$value");
                [ "$option" = "h" ] && shift;;
            r | rank_table_file)
                check_empty "$option" "$value"; RANK_TABLE_FILE=$(realpath -e "$value");
                [ "$option" = "r" ] && shift;;
            w | weights)
                check_empty "$option" "$value"; WEIGHTS=$(realpath -e "$value");
                [ "$option" = "w" ] && shift;;
            m | mindir)
                check_empty "$option" "$value"; MINDIR=$(realpath -e "$value");
                [ "$option" = "m" ] && shift;;
            c | config)
                check_empty "$option" "$value"; CONFIG_PATH=$(realpath -e "$value");
                [ "$option" = "c" ] && shift;;
            D | device)
                check_empty "$option" "$value"; check_number "$option" "$value"; DEVICE_ID=$value;
                [ "$option" = "D" ] && shift;;
            n | number)
                check_empty "$option" "$value"; check_number "$option" "$value"; DEVICE_NUM=$value;
                [ "$option" = "n" ] && shift;;
            *)  echo "Unknown option: $1. Please see usage:"; usage; exit 1;;
        esac
        shift
    done
}


function get_default_config() {
    if [ -z "$CONFIG_PATH" ]; then
        CONFIG_PATH=$(realpath -e "../config/network/yolov7.yaml")
        echo "Use default config file $CONFIG_PATH because CONFIG_PATH argument is empty."
    fi
    if [ -z "$DATA_PATH" ]; then
        DATA_PATH=$(realpath -e "../config/data/coco.yaml")
        echo "Use default config file $DATA_PATH because DATA_PATH argument is empty."
    fi
    if [ -z "$HYP_PATH" ]; then
        HYP_PATH=$(realpath -e "../config/data/hyp.scratch.p5.yaml")
        echo "Use default config file $HYP_PATH because HYP_PATH argument is empty."
    fi
    if [ -z "$DEVICE_ID" ]; then
        DEVICE_ID=0
        echo "Use default device id $DEVICE_ID because DEVICE_ID argument is empty."
    fi
    if [ -z "$DEVICE_NUM" ]; then
        DEVICE_NUM=8
        echo "Use default device number $DEVICE_ID because DEVICE_NUM argument is empty."
    fi
}