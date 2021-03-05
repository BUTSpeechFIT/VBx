#!/bin/bash

INSTRUCTION=$1
METHOD=$2 # AHC or AHC+VB

exp_dir=$3 # output experiment directory
xvec_dir=$4 # output xvectors directory
WAV_DIR=$5 # wav files directory
FILE_LIST=$6 # txt list of files to process
LAB_DIR=$7 # lab files directory with VAD segments
RTTM_DIR=$8 # reference rttm files directory

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


if [[ $INSTRUCTION = "xvectors" ]]; then
	WEIGHTS_DIR=$DIR/VBx/models/ResNet101_16kHz/nnet
	if [ ! -f $WEIGHTS_DIR/raw_81.pth ]; then
	    cat $WEIGHTS_DIR/raw_81.pth.zip.part* > $WEIGHTS_DIR/unsplit_raw_81.pth.zip
		unzip $WEIGHTS_DIR/unsplit_raw_81.pth.zip -d $WEIGHTS_DIR/
	fi

	WEIGHTS=$DIR/VBx/models/ResNet101_16kHz/nnet/raw_81.pth
	EXTRACT_SCRIPT=$DIR/VBx/extract.sh
	DEVICE=cpu

	mkdir -p $xvec_dir
	$EXTRACT_SCRIPT ResNet101 $WEIGHTS $WAV_DIR $LAB_DIR $FILE_LIST $xvec_dir $DEVICE

	# Replace this to submit jobs to a grid engine
	bash $xvec_dir/xv_task
fi


BACKEND_DIR=$DIR/VBx/models/ResNet101_16kHz
if [[ $INSTRUCTION = "diarization" ]]; then
	TASKFILE=$exp_dir/diar_"$METHOD"_task
	OUTFILE=$exp_dir/diar_"$METHOD"_out
	rm -f $TASKFILE $OUTFILE
	mkdir -p $exp_dir/lists

	thr=-0.015
	smooth=7.0
	lda_dim=128
	Fa=0.2
	Fb=6
	loopP=0.35
	OUT_DIR=$exp_dir/out_dir_"$METHOD"
	if [[ ! -d $OUT_DIR ]]; then
		mkdir -p $OUT_DIR
		while IFS= read -r line; do
			grep $line $FILE_LIST > $exp_dir/lists/$line".txt"
			python3="unset PYTHONPATH ; unset PYTHONHOME ; export PATH=\"/mnt/matylda5/iplchot/python_public/anaconda3/bin:$PATH\""
			echo "$python3 ; python $DIR/VBx/vbhmm.py --init $METHOD --out-rttm-dir $OUT_DIR/rttms --xvec-ark-file $xvec_dir/xvectors/$line.ark --segments-file $xvec_dir/segments/$line --plda-file $BACKEND_DIR/plda --xvec-transform $BACKEND_DIR/transform.h5 --threshold $thr --init-smoothing $smooth --lda-dim $lda_dim --Fa $Fa --Fb $Fb --loopP $loopP" >> $TASKFILE
		done < $FILE_LIST
		bash $TASKFILE > $OUTFILE

		# Score
		cat $OUT_DIR/rttms/*.rttm > $OUT_DIR/sys.rttm
		cat $RTTM_DIR/*.rttm > $OUT_DIR/ref.rttm
		$DIR/dscore/score.py --collar 0.25 -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm > $OUT_DIR/result_fair
		$DIR/dscore/score.py --collar 0.0 -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm > $OUT_DIR/result_full
	fi
fi