#!/bin/bash

SET=$1 # dev or eval
INSTRUCTION=$2
exp_dir=$3 # output experiment directory
WAV_DIR=$4 # wav files directory
RTTM_DIR=$5 # reference rttm files directory (from downloaded dev set)

USE_FINAL_VAD=TRUE # if FALSE, compute and use energy VAD. If TRUE, use the shared VAD files

if [ $# -lt 4 ]; then
    echo "The set, instruction, output directory and waveform directory must be given."
    exit 1
fi

if [[ "$INSTRUCTION" != "VAD" ]] && [[ "$INSTRUCTION" != "xvectors" ]] && [[ "$INSTRUCTION" != "VBx" ]] && [[ "$INSTRUCTION" != "score" ]] && [[ "$INSTRUCTION" != "global_xvectors" ]] && [[ "$INSTRUCTION" != "recluster" ]] && [[ "$INSTRUCTION" != "score_recluster" ]] && [[ "$INSTRUCTION" != "OV_heuristic" ]] && [[ "$INSTRUCTION" != "score_heuristic" ]] && [[ "$INSTRUCTION" != "OV_label2nd" ]] && [[ "$INSTRUCTION" != "score_label2nd" ]] && [[ "$INSTRUCTION" != "global_xvectors" ]]; then
	echo "Incorrect instruction"
    exit 1
fi


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


WEIGHTS=$DIR/VBx/models/ResNet152_16kHz/nnet/raw_58.pth
BACKEND_DIR=$DIR/VBx/models/ResNet152_16kHz
FILE_LIST=$DIR/$SET.txt

if [[ $INSTRUCTION = "VAD" ]]; then
	if [[ $USE_FINAL_VAD = "FALSE" ]]; then
		vad_dir=$exp_dir/energy_VAD/labs
		python $DIR/VAD/energy_VAD.py --in-audio-dir $WAV_DIR --vad-out-dir $vad_dir \
				--list $FILE_LIST --threshold 0.75 --median-window-length 85
	else
		echo "Nothing to do, the VAD is already computed."
	fi
fi


if [[ $INSTRUCTION = "xvectors" ]]; then
	if [[ $USE_FINAL_VAD = "FALSE" ]]; then
		vad_dir=$exp_dir/energy_VAD/labs
		xvec_dir=$exp_dir/energy_VAD/xvectors
	else
		vad_dir=$DIR/VAD/final_system/labs_$SET
		xvec_dir=$exp_dir/final_VAD/xvectors
	fi

	EXTRACT_SCRIPT=$DIR/VBx/extract.sh
	DEVICE=cpu

	mkdir -p $xvec_dir
	# This script creates the TASKFILE
	$EXTRACT_SCRIPT ResNet152 $WEIGHTS $WAV_DIR $vad_dir $FILE_LIST $xvec_dir $DEVICE

	# Replace this to submit jobs to a grid engine
	bash $xvec_dir/xv_task
fi


if [[ $INSTRUCTION = "VBx" ]]; then
	if [[ $USE_FINAL_VAD = "FALSE" ]]; then
		xvec_dir=$exp_dir/energy_VAD/xvectors
		VBx_dir=$exp_dir/energy_VAD/VBx
	else
		xvec_dir=$exp_dir/final_VAD/xvectors
		VBx_dir=$exp_dir/final_VAD/VBx
	fi

	TASKFILE=$VBx_dir/diar_task
	OUTFILE=$VBx_dir/diar_out
	rm -f $TASKFILE $OUTFILE

	thr=1.0
	tareng=0.55
	smooth=5.0
	lda_dim=256
	Fa=0.3
	Fb=14
	loopP=0.9
	if [[ ! -d $VBx_dir ]]; then
		mkdir -p $VBx_dir/lists
		while IFS= read -r line; do
			grep $line $FILE_LIST > $VBx_dir/lists/$line".txt"
			echo "python $DIR/VBx/vbhmm.py --init AHC+VB --out-rttm-dir $VBx_dir/rttms --output-2nd True --xvec-ark-file $xvec_dir/xvectors/$line.ark --segments-file $xvec_dir/segments/$line --plda-file $BACKEND_DIR/plda --xvec-tran $BACKEND_DIR/transform.mat --xvec-mean $BACKEND_DIR/mean.vec --threshold $thr --target-energy $tareng --init-smoothing $smooth --lda-dim $lda_dim --Fa $Fa --Fb $Fb --loopP $loopP" >> $TASKFILE
		done < $FILE_LIST

		# Replace this to submit jobs to a grid engine
		bash $TASKFILE > $OUTFILE
	fi
fi


if [[ $INSTRUCTION = "score" ]]; then
	if [[ $USE_FINAL_VAD = "FALSE" ]]; then
		VBx_dir=$exp_dir/energy_VAD/VBx
	else
		VBx_dir=$exp_dir/final_VAD/VBx
	fi

	if [[ $SET = "dev" ]]; then
		cat $VBx_dir/rttms/*.rttm > $VBx_dir/sys.rttm
		cat $RTTM_DIR/*.rttm > $VBx_dir/ref.rttm
		$DIR/dscore/score.py --collar 0.25 -r $VBx_dir/ref.rttm -s $VBx_dir/sys.rttm > $VBx_dir/result
	else
		echo "Score only available for dev set."
	fi
fi


if [[ $INSTRUCTION = "global_xvectors" ]]; then
	if [[ $USE_FINAL_VAD = "FALSE" ]]; then
		VBx_dir=$exp_dir/energy_VAD/VBx
		xvec_per_speaker_dir=$exp_dir/energy_VAD/xvectors_per_speaker
	else
		xvec_per_speaker_dir=$exp_dir/final_VAD/xvectors_per_speaker
		VBx_dir=$exp_dir/final_VAD/VBx
	fi

	EXTRACT_SCRIPT=$DIR/VBx/extract_global_per_speaker.sh
	DEVICE=cpu

	mkdir -p $xvec_per_speaker_dir
	# This script creates the TASKFILE
	$EXTRACT_SCRIPT ResNet152 $WEIGHTS $WAV_DIR $VBx_dir/rttms $FILE_LIST $xvec_per_speaker_dir $DEVICE

	TASKFILE=$xvec_per_speaker_dir/xv_global_task
	OUTFILE=$xvec_per_speaker_dir/xv_global_out

	# Replace this to submit jobs to a grid engine
	bash $TASKFILE > $OUTFILE
fi


if [[ $INSTRUCTION = "recluster" ]]; then
	if [[ $USE_FINAL_VAD = "FALSE" ]]; then
		VBx_dir=$exp_dir/energy_VAD/VBx
		xvec_per_speaker_dir=$exp_dir/energy_VAD/xvectors_per_speaker
		recluster_dir=$exp_dir/energy_VAD/recluster
	else
		VBx_dir=$exp_dir/final_VAD/VBx
		xvec_per_speaker_dir=$exp_dir/final_VAD/xvectors_per_speaker
		recluster_dir=$exp_dir/final_VAD/recluster
	fi

	TASKFILE=$recluster_dir/recluster_task
	OUTFILE=$recluster_dir/recluster_out
	rm -f $TASKFILE $OUTFILE

	mkdir -p $recluster_dir/rttms $recluster_dir/rttms2nd
	echo "python $DIR/VBx/recluster.py --in-file-list $FILE_LIST --in-rttm-dir $VBx_dir/rttms --in-rttm-dir2 $VBx_dir/rttms2nd --in-ark-dir $xvec_per_speaker_dir/xvectors --out-rttm-dir $recluster_dir/rttms --out-rttm-dir2 $recluster_dir/rttms2nd --mean-vec-file $BACKEND_DIR/mean.vec --tran-mat-file $BACKEND_DIR/transform.mat --plda-file $BACKEND_DIR/plda --tar-eng 1.0 --threshold -0.5" >> $TASKFILE
	
	bash $TASKFILE > $OUTFILE
fi


if [[ $INSTRUCTION = "score_recluster" ]]; then
	if [[ $USE_FINAL_VAD = "FALSE" ]]; then
		recluster_dir=$exp_dir/energy_VAD/recluster
	else
		recluster_dir=$exp_dir/final_VAD/recluster
	fi

	if [[ $SET = "dev" ]]; then
		cat $recluster_dir/rttms/*.rttm > $recluster_dir/sys_reclustering.rttm
		cat $RTTM_DIR/*.rttm > $recluster_dir/ref.rttm
		$DIR/dscore/score.py --collar 0.25 -r $recluster_dir/ref.rttm -s $recluster_dir/sys_reclustering.rttm > $recluster_dir/result_reclustering
	else
		echo "Score only available for dev set."
	fi
fi


if [[ $INSTRUCTION = "OV_heuristic" ]]; then
	if [[ $USE_FINAL_VAD = "FALSE" ]]; then
		recluster_dir=$exp_dir/energy_VAD/recluster
		OV_heuristic_dir=$exp_dir/energy_VAD/OV_heuristic
	else
		recluster_dir=$exp_dir/final_VAD/recluster
		OV_heuristic_dir=$exp_dir/final_VAD/OV_heuristic
	fi
	
	TASKFILE=$OV_heuristic_dir/heuristic_task
	OUTFILE=$OV_heuristic_dir/heuristic_out
	rm -f $TASKFILE $OUTFILE

	mkdir -p $OV_heuristic_dir/rttms
	echo "python $DIR/OVD/handling.py --in-file-list $FILE_LIST --in-rttm-dir $recluster_dir/rttms --in-rttm-with-ov-dir $DIR/OVD/final_system/rttms_$SET --out-rttm-dir $OV_heuristic_dir/rttms --heuristic True" >> $TASKFILE
	
	# Replace this to submit jobs to a grid engine
	bash $TASKFILE > $OUTFILE
fi


if [[ $INSTRUCTION = "score_heuristic" ]]; then
	if [[ $USE_FINAL_VAD = "FALSE" ]]; then
		OV_heuristic_dir=$exp_dir/energy_VAD/OV_heuristic
	else
		OV_heuristic_dir=$exp_dir/final_VAD/OV_heuristic
	fi
	if [[ $SET = "dev" ]]; then
		cat $OV_heuristic_dir/rttms/*.rttm > $OV_heuristic_dir/sys_heuristic.rttm
		cat $RTTM_DIR/*.rttm > $OV_heuristic_dir/ref.rttm
		$DIR/dscore/score.py --collar 0.25 -r $OV_heuristic_dir/ref.rttm -s $OV_heuristic_dir/sys_heuristic.rttm > $OV_heuristic_dir/result_heuristic
	else
		echo "Score only available for dev set."
	fi
fi


if [[ $INSTRUCTION = "OV_label2nd" ]]; then
	if [[ $USE_FINAL_VAD = "FALSE" ]]; then
		recluster_dir=$exp_dir/energy_VAD/recluster
		OV_label2nd_dir=$exp_dir/energy_VAD/OV_label2nd
	else
		recluster_dir=$exp_dir/final_VAD/recluster
		OV_label2nd_dir=$exp_dir/final_VAD/OV_label2nd
	fi
	
	TASKFILE=$OV_label2nd_dir/label2nd_task
	OUTFILE=$OV_label2nd_dir/label2nd_out
	rm -f $TASKFILE $OUTFILE

	mkdir -p $OV_label2nd_dir/rttms
	echo "python $DIR/OVD/handling.py --in-file-list $FILE_LIST --in-rttm-dir $recluster_dir/rttms --in-2nd-rttm-dir $recluster_dir/rttms2nd --in-rttm-with-ov-dir $DIR/OVD/final_system/rttms_$SET --out-rttm-dir $OV_label2nd_dir/rttms --label2nd True" >> $TASKFILE
	
	# Replace this to submit jobs to a grid engine
	bash $TASKFILE > $OUTFILE
fi


if [[ $INSTRUCTION = "score_label2nd" ]]; then
	if [[ $USE_FINAL_VAD = "FALSE" ]]; then
		OV_label2nd_dir=$exp_dir/energy_VAD/OV_label2nd
	else
		OV_label2nd_dir=$exp_dir/final_VAD/OV_label2nd
	fi
	
	if [[ $SET = "dev" ]]; then
		cat $OV_label2nd_dir/rttms/*.rttm > $OV_label2nd_dir/sys_label2nd.rttm
		cat $RTTM_DIR/*.rttm > $OV_label2nd_dir/ref.rttm
		$DIR/dscore/score.py --collar 0.25 -r $OV_label2nd_dir/ref.rttm -s $OV_label2nd_dir/sys_label2nd.rttm > $OV_label2nd_dir/result_label2nd
	else
		echo "Score only available for dev set."
	fi
fi
