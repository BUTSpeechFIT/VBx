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


import argparse
from shutil import copyfile
import numpy as np
import scipy.linalg as spl
import kaldi_io
from kaldi_utils import read_plda
from diarization_lib import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--in-file-list', required=True, type=str, help='list of files')
	parser.add_argument('--in-rttm-dir', required=True, type=str, help='input rttm directory')
	parser.add_argument('--in-rttm-dir2', required=False, type=str, help='input rttm directory for second speaker')
	parser.add_argument('--in-ark-dir', required=True, type=str, help='input ark directory with global xvectors')
	parser.add_argument('--out-rttm-dir', required=True, type=str, help='output rttm directory')
	parser.add_argument('--out-rttm-dir2', required=False, type=str, help='output rttm directory for second speaker')
	parser.add_argument('--mean-vec-file', required=True, type=str, help='path to x-vector mean file')
	parser.add_argument('--tran-mat-file', required=True, type=str, help='path to x-vector transformation file')
	parser.add_argument('--plda-file', required=True, type=str, help='File with PLDA model in Kaldi format used for AHC')
	parser.add_argument('--tar-eng', required=True, type=str, help='target energy to keep after PCA')
	parser.add_argument('--threshold', required=True, type=str, help='AHC threshold')

	args = parser.parse_args()

	threshold = float(args.threshold)
	target_energy = float(args.tar_eng)
	file_names = np.atleast_1d(np.loadtxt(args.in_file_list, dtype=object))

	glob_mean = kaldi_io.read_vec_flt(args.mean_vec_file) # x-vector centering vector
	glob_tran = kaldi_io.read_mat(args.tran_mat_file) # x-vector whitening transformation
	glob_tran = glob_tran[:,0:-1] # Skipping last column
	plda = read_plda(args.plda_file)
	plda_mu, plda_tr, plda_psi = plda

	W = np.linalg.inv(plda_tr.T.dot(plda_tr))
	B = np.linalg.inv((plda_tr.T/plda_psi).dot(plda_tr))
	acvar, wccn = spl.eigh(B,  W)
	plda_psi = acvar[::-1]
	plda_tr = wccn.T[::-1]

	for fn in file_names:
		print(fn)

		segs = np.loadtxt(args.in_rttm_dir+'/'+fn+'.rttm', usecols=[3,4], ndmin=1) #each row is a turn, columns denote beggining of the turn (s) and duration (s)
		spks = np.loadtxt(args.in_rttm_dir+'/'+fn+'.rttm', usecols=[7],dtype='str', ndmin=1) #spk id of each turn
		if len(set(spks)) == 1:
			copyfile(args.in_rttm_dir+'/'+fn+'.rttm', args.out_rttm_dir+'/'+fn+'.rttm')
			if args.in_rttm_dir2 is not None:
				copyfile(args.in_rttm_dir2+'/'+fn+'.rttm', args.out_rttm_dir2+'/'+fn+'.rttm')
		else:
			# Read all xvectors and compare them
			speaker_positions = {}
			xvectors = []
			i = 0
			for speaker in set(spks):
				arkit = kaldi_io.read_vec_flt_ark(args.in_ark_dir+'/'+fn+'_'+str(speaker)+'.ark')
				pair = [v for v in arkit][0]
				x = pair[1]
				speaker_positions[i] = pair[0]
				i += 1
				xvectors.append(x)
			x = np.vstack(xvectors)
			# Kaldi-like global norm and length-norm
			x = (x-glob_mean).dot(glob_tran.T)
			x *= np.sqrt(x.shape[1] / (x**2).sum(axis=1)[:,np.newaxis])
			scr_mx = kaldi_ivector_plda_scoring_dense((plda_mu, plda_tr, plda_psi), x, target_energy=target_energy)
			thr, junk = twoGMMcalib_lin(scr_mx.ravel()) # Optionally, figure out utterance specific threshold for AHC.
			labels = AHC(scr_mx, thr+threshold) # output "labels" is integer vector of speaker (cluster) ids
			
			# Map input speaker labels to labels after reclustering
			map_dict = {}
			for i in range(labels.shape[0]):
				map_dict[speaker_positions[i]] = labels[i]
			# Write segments with speakers remapped
			with open(args.out_rttm_dir+'/'+fn+'.rttm', 'w') as fp:
				for i in range(segs.shape[0]):
					fp.write("SPEAKER %s 1 %.3f %.3f <NA> <NA> %d <NA> <NA>\n" % (fn, segs[i][0], segs[i][1], map_dict[fn+'_'+spks[i]]))

			if args.in_rttm_dir2 is not None:
				segs = np.loadtxt(args.in_rttm_dir2+'/'+fn+'.rttm', usecols=[3,4], ndmin=1) #each row is a turn, columns denote beggining of the turn (s) and duration (s)
				spks = np.loadtxt(args.in_rttm_dir2+'/'+fn+'.rttm', usecols=[7],dtype='str', ndmin=1) #spk id of each turn
				# Write segments with speakers remapped
				with open(args.out_rttm_dir2+'/'+fn+'.rttm', 'w') as fp:
					for i in range(segs.shape[0]):
						fp.write("SPEAKER %s 1 %.3f %.3f <NA> <NA> %d <NA> <NA>\n" % (fn, segs[i][0], segs[i][1], map_dict[fn+'_'+spks[i]]))
		