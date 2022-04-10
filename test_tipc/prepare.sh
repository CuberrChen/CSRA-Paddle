#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
# MODE be one of ['lite_train_lite_infer']
MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

# MODE be one of ['lite_train_lite_infer']
if [ ${MODE} = "lite_train_lite_infer" ];then
    rm -r ./test_tipc/data/VOCdevkit
    cd ./test_tipc/data/ && unzip VOC2007_Small.zip && cd ../../
    python utils/prepare/prepare_voc.py  --data_path test_tipc/data/VOCdevkit
fi