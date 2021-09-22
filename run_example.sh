#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p exp

for audio in $(ls example/audios/16k)
do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list.txt

      # run feature and x-vectors extraction
      python VBx/predict.py \
          --in-file-list exp/list.txt \
          --in-lab-dir example/vad \
          --in-wav-dir example/audios/16k \
          --out-ark-fn exp/${filename}.ark \
          --out-seg-fn exp/${filename}.seg \
          --weights VBx/models/ResNet101_16kHz/nnet/final.onnx \
          --backend onnx

      # run variational bayes on top of x-vectors
      python VBx/vbhmm.py \
          --init AHC+VB \
          --out-rttm-dir exp \
          --xvec-ark-file exp/${filename}.ark \
          --segments-file exp/${filename}.seg \
          --xvec-transform VBx/models/ResNet101_16kHz/transform.h5 \
          --plda-file VBx/models/ResNet101_16kHz/plda \
          --threshold -0.015 \
          --lda-dim 128 \
          --Fa 0.3 \
          --Fb 17 \
          --loopP 0.99

      # check if there is ground truth .rttm file
      if [ -f example/rttm/${filename}.rttm ]
      then
          # run dscore
          python dscore/score.py -r example/rttm/${filename}.rttm -s exp/${filename}.rttm --collar 0.25 --ignore_overlaps
      fi
done