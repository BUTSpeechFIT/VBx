#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# @Authors: Federico Landini
# @Emails: landini@fit.vutbr.cz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


MODEL=$1
WEIGHTS=$2
WAV_DIR=$3
RTTM_DIR=$4
FILE_LIST=$5
OUT_DIR=$6
DEVICE=$7

EMBED_DIM=256
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -pv $OUT_DIR

TASKFILE=$OUT_DIR/xv_global_task
rm -f $TASKFILE

mkdir -p $OUT_DIR/lists $OUT_DIR/xvectors $OUT_DIR/segments
while IFS= read -r line; do
    mkdir -p "$(dirname $OUT_DIR/lists/$line)"
    grep $line $FILE_LIST > $OUT_DIR/lists/$line".txt"
    OUT_ARK_FILE=$OUT_DIR/xvectors/$line.ark
    OUT_SEG_FILE=$OUT_DIR/segments/$line
    mkdir -p "$(dirname $OUT_ARK_FILE)"
    mkdir -p "$(dirname $OUT_SEG_FILE)"
    if [[ "$DEVICE" == "gpu" ]]; then
        echo "python $DIR/predict_global_xvectors.py --seg-len 144 --seg-jump 24 --model $MODEL --weights $WEIGHTS --gpus=\$($DIR/free_gpu.sh) $MDL_WEIGHTS --ndim 64 --embed-dim $EMBED_DIM --in-file-list $OUT_DIR/lists/$line".txt" --in-rttm-dir $RTTM_DIR --in-wav-dir $WAV_DIR --out-ark-fn $OUT_ARK_FILE --out-seg-fn $OUT_SEG_FILE" >> $TASKFILE
    else
        echo "python $DIR/predict_global_xvectors.py --seg-len 144 --seg-jump 24 --model $MODEL --weights $WEIGHTS --gpus= $MDL_WEIGHTS --ndim 64 --embed-dim $EMBED_DIM --in-file-list $OUT_DIR/lists/$line".txt" --in-rttm-dir $RTTM_DIR --in-wav-dir $WAV_DIR --out-ark-fn $OUT_ARK_FILE --out-seg-fn $OUT_SEG_FILE" >> $TASKFILE
    fi
done < $FILE_LIST
