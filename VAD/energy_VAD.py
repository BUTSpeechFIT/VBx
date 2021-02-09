#!/usr/bin/env python

# Copyright Brno University of Technology
# Licensed under the Apache License, Version 2.0 (the "License")

import numpy as np
import sys, os, errno
import argparse
import gmm
import soundfile as sf
from scipy.io import wavfile
from scipy.special import logsumexp


def add_dither(x, level=8):
    return x + level * (np.random.rand(*x.shape)*2-1)


def compute_vad(s, win_length=160, win_overlap=80, n_realignment=5, threshold=0.3):
	# power signal for energy computation
	s = s**2
	# frame signal with overlap
	F = framing(s, win_length, win_length - win_overlap) 
	# sum frames to get energy
	E = F.sum(axis=1).astype(np.float64)

	# normalize the energy
	E -= E.mean()
	try:
		E /= E.std()
		# initialization
		mm = np.array((-1.00, 0.00, 1.00))[:, np.newaxis]
		ee = np.array(( 1.00, 1.00, 1.00))[:, np.newaxis]
		ww = np.array(( 0.33, 0.33, 0.33))

		GMM = gmm.gmm_eval_prep(ww, mm, ee)

		E = E[:,np.newaxis]

		for i in range(n_realignment):
			# collect GMM statistics
			llh, N, F, S = gmm.gmm_eval(E, GMM, return_accums=2)
			# update model
			ww, mm, ee   = gmm.gmm_update(N, F, S)
			# wrap model
			GMM = gmm.gmm_eval_prep(ww, mm, ee)

		# evaluate the gmm llhs
		llhs = gmm.gmm_llhs(E, GMM)
		llh  = logsumexp(llhs, axis=1)[:,np.newaxis]
		llhs = np.exp(llhs - llh)

		out  = np.zeros(llhs.shape[0], dtype=np.bool)
		out[llhs[:,0] < threshold] = True
	except RuntimeWarning:
		logging.info("File contains only silence")
		out=np.zeros(E.shape[0],dtype=np.bool)
	return out


def frame_labels2start_ends(speech_frames, frame_rate=100.0):
    decesions = np.r_[False, speech_frames, False]
    return np.nonzero(decesions[1:] != decesions[:-1])[0].reshape(-1,2) / frame_rate

    
def framing(a, window, shift=1):
    shape = ((a.shape[0] - window) // shift + 1, window) + a.shape[1:]
    strides = (a.strides[0]*shift,a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def median_filter(vad, win_length=50, threshold=25):
	stats = np.r_[vad, np.zeros(win_length)].cumsum()-np.r_[np.zeros(win_length), vad].cumsum()
	return stats[win_length//2:win_length//2-win_length] > threshold


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--in-audio-dir', type=str, help="Input audio directory")
	parser.add_argument('--vad-out-dir', type=str, help="Output lab directory")
	parser.add_argument("--list", type=str, help="txt list of files")
	parser.add_argument('--in-format', type=str, help="Input format")
	parser.add_argument("--threshold", type=float, default=0.5)
	parser.add_argument("--median-window-length", type=int, default=50)

	args = parser.parse_args()

	txt_list = [ line.rstrip() for line in open(args.list,'r') ]
	for key in txt_list:
		temp_file = os.path.join(args.vad_out_dir+"/"+key+".lab_part")
		mkdir_p(os.path.dirname(args.vad_out_dir+"/"+key+".lab"))
		vad_file = os.path.join(args.vad_out_dir+"/"+key+".lab")
		samplerate, signal = wavfile.read(args.in_audio_dir+"/"+key+".wav")
		
		signal = np.r_[np.zeros(samplerate//2), signal, np.zeros(samplerate//2)] # add half second of "silence" at the beginning and the end
		np.random.seed(3)  # for reproducibility
		signal = add_dither(signal, 8.0)
		vad = compute_vad(signal, win_length=int(round(0.025*samplerate)), win_overlap=int(round(0.015*samplerate)), n_realignment=5, threshold=args.threshold)
		vad = median_filter(vad, win_length=args.median_window_length, threshold=5)
		labels = frame_labels2start_ends(vad[50:-50], frame_rate=1.0)/100.0
		np.savetxt(temp_file, np.array(labels, dtype=object), fmt='%.3f\t%.3f\tspeech')
		os.rename(temp_file, vad_file)


if __name__ == '__main__':
    main()
