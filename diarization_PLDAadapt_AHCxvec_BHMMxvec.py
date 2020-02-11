#!/usr/bin/env python

# Copyright 2019 Lukas Burget (burget@fit.vutbr.cz)
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


# Recipe for doing diarization on data from The Second DIHARD Diarization Challenge
# https://coml.lscp.ens.fr/dihard/index.html
# The recipe consists in doing Agglomerative Hierachical Clustering on
# x-vectors in a first step. Then, Variational Bayes HMM over x-vectors
# is applied using the AHC output as initialization.
# 
# The BUT submission for the challenge is presented in
# F. Landini, S. Wang, M. Diez, L. Burget et al.
# BUT System for the Second DIHARD Speech Diarization Challenge, ICASSP 2020
# and a more detailed analysis of this approach is presented in 
# M. Diez, L. Burget, F. Landini, S. Wang, J. \v{C}ernock\'{y}
# Optimizing Bayesian HMM based x-vector clustering for the second DIHARD speech 
# diarization challenge, ICASSP 2020

# A more thorough description and study of the VB-HMM with eigen-voice priors 
# approach for diarization is presented in 
# M. Diez, L. Burget, F. Landini, J. \v{C}ernock\'{y}
# Analysis of Speaker Diarization based on Bayesian HMM with Eigenvoice Priors, 
# IEEE Transactions on Audio, Speech and Language Processing, 2019

# This recipe differs from our submission to the challenge in that
# VB resegmentation and overlapped speech post-processing are not applied.
# These two steps are not presented for producing small improvements
# but adding considerably more complicated processing to the recipe.

import sys
import numpy as np
import itertools
import kaldi_io
from diarization_lib import *
import VB_diarization
import time
from scipy.special import softmax

out_rttm_dir  =       sys.argv[1]   # Directory to store output rttm files
xvec_ark_file =       sys.argv[2]   # Kaldi ark file with x-vectors from one or more input recordings 
                                    # Attention: all x-vectors from one recording must be in one ark file
segments_file =       sys.argv[3]   # File with x-vector timing info (see diarization_lib.read_xvector_timing_dict)
mean_vec_file =       sys.argv[4]   # File with mean vector in Kaldi format for x-vector centering
tran_mat_file =       sys.argv[5]   # File with linear transformation matrix in Kaldi format for x-vector whitening
plda_file     =       sys.argv[6]   # File with PLDA model in Kaldi format used for AHC and VB-HMM x-vector clustering
plda_adapt_file=      sys.argv[7]   # Another PLDA model in Kaldi format which is interpolated with the previous one
alpha         = float(sys.argv[8])  # Interpolation weight between 0 and 1 for mixing the two PLDA model alpha=0 corresponds to plda_adapt
threshold     = float(sys.argv[9])  # Threshold (bias) used for AHC
target_energy = float(sys.argv[10]) # Parameter affecting AHC. (see diarization_lib.kaldi_ivector_plda_scoring_dense)
init_smoothing= float(sys.argv[11]) # AHC produces hard assignments of x-vetors to speakers. These are "smoothed" to
                                    # soft assignments as the initialization for VB-HMM. This parameter controls the amount
                                    # of smoothing. Not so important, high value (e.g. 10) is OK  => keeping hard assigment
lda_dim       =   int(sys.argv[12]) # For VB-HMM, x-vectors are reduced to this dimensionality using LDA
Fa            = float(sys.argv[13]) # Parameter of VB-HMM (see VB_diarization.VB_diarization)
Fb            = float(sys.argv[14]) # Parameter of VB-HMM (see VB_diarization.VB_diarization)
LoopP         = float(sys.argv[15]) # Parameter of VB-HMM (see VB_diarization.VB_diarization)
use_VB        = True                # False for using only AHC

frm_shift = 0.01 # frame rate of MFCC features

glob_tran = kaldi_io.read_mat(tran_mat_file)           # x-vector whitening transformation
glob_mean = kaldi_io.read_vec_flt(mean_vec_file)       # x-vector centering vector
kaldi_plda_train = kaldi_io.read_plda(plda_file)       # out-of-domain PLDA model
kaldi_plda_adapt = kaldi_io.read_plda(plda_adapt_file) # in-domain "adaptation" PLDA model
segs_dict = read_xvector_timing_dict(segments_file)    # segments file with x-vector timing information

plda_train_mu, plda_train_tr, plda_train_psi = kaldi_plda_train
plda_adapt_mu, plda_adapt_tr, plda_adapt_psi = kaldi_plda_adapt

# Interpolate across-class, within-class and means of the two PLDA models with interpolation factor "alpha"
plda_mu = alpha*plda_train_mu + (1.0-alpha)*plda_adapt_mu
W_train = np.linalg.inv(plda_train_tr.T.dot(plda_train_tr))
B_train = np.linalg.inv((plda_train_tr.T/plda_train_psi).dot(plda_train_tr))
W_adapt = np.linalg.inv(plda_adapt_tr.T.dot(plda_adapt_tr))
B_adapt = np.linalg.inv((plda_adapt_tr.T/plda_adapt_psi).dot(plda_adapt_tr))
W = alpha * W_train + (1.0-alpha) * W_adapt
B = alpha * B_train + (1.0-alpha) * B_adapt
acvar, wccn = spl.eigh(B,  W)
plda_psi = acvar[::-1]
plda_tr = wccn.T[::-1]

# Prepare model for VB-HMM clustering (see the comment on "fea" variable below)
ubmWeights = np.array([1.0])
ubmMeans = np.zeros((1,lda_dim))
invSigma= np.ones((1,lda_dim))
V=np.diag(np.sqrt(plda_psi[:lda_dim]))[:,np.newaxis,:]
VtinvSigmaV = VB_diarization.precalculate_VtinvSigmaV(V, invSigma)

# Open ark file with x-vectors and in each iteration of the following for-loop
# read a batch of x-vectors corresponding to one recording
arkit = kaldi_io.read_vec_flt_ark(xvec_ark_file)
recit = itertools.groupby(arkit, lambda e: e[0].rsplit('_', 1)[0]) # group xvectors in ark by recording name
for file_name, segs in recit:
    print(file_name)

    seg_names, xvecs = zip(*segs)
    x = np.array(xvecs) # matrix of all x-vectors corresponding to recording "file_name"

    #hac_start = time.time()
    # Kaldi-like global norm and lenth-norm
    x = (x-glob_mean).dot(glob_tran.T)
    x *= np.sqrt(x.shape[1] / (x**2).sum(axis=1)[:,np.newaxis])

    # Kaldi-like AHC of x-vectors (scr_mx is matrix of pairwise similarities between all x-vectors)
    scr_mx = kaldi_ivector_plda_scoring_dense((plda_mu, plda_tr, plda_psi), x, target_energy=target_energy)
    thr, junk = twoGMMcalib_lin(scr_mx.ravel()) # Optionally, figure out utterance specific threshold for AHC.
    labels = AHC(scr_mx, thr+threshold) # output "labels" is integer vector of speaker (cluster) ids

    #hac_time = time.time()-hac_start

    if use_VB:
        #vbx_start = time.time()

        # Smooth the hard labels obtained from AHC to soft assignments of x-vectors to speakers
        q_init = np.zeros((len(labels), np.max(labels)+1))
        q_init[range(len(labels)), labels] = 1.0
        q_init = softmax(q_init*init_smoothing, axis=1)

        # Transform x-vectors to LDA space and reduce its dimensionality
        # Now, mean is 0, within-class covariance identity and across-class covariance  diagonal (plda_psi)
        fea = (x-plda_mu).dot(plda_tr.T)[:,:lda_dim]

        # Use VB-HMM for x-vector clustering. Instead of i-vector extractor model, we use PLDA
        # => GMM with only 1 component, V derived accross-class covariance, and invSigma is inverse within-class covariance (i.e. identity)
        q, sp, L = VB_diarization.VB_diarization(fea, ubmMeans, invSigma, ubmWeights, V, pi=None, gamma=q_init, maxSpeakers=q_init.shape[1], maxIters=40, VtinvSigmaV=VtinvSigmaV,
                                        downsample=None, sparsityThr=0.001, epsilon=1e-6, loopProb=LoopP, Fa=Fa, Fb=Fb)

        #vbx_time = time.time()-vbx_start

        labels = np.unique(q.argmax(1), return_inverse=True)[1] 

    assert(np.all(segs_dict[file_name][0] == np.array(seg_names)))
    start, end = segs_dict[file_name][1].T
    starts, ends, out_labels  = merge_adjacent_labels(start, end, labels)

#######################################################################################
    mkdir_p(out_rttm_dir)
    with open(out_rttm_dir+'/'+file_name+'.rttm', 'w') as fp:
      for l, s, e in zip(out_labels, starts, ends):
        fp.write("SPEAKER %s 1 %.3f %.3f <NA> <NA> %d <NA> <NA>\n" % (file_name, s, e-s, l+1))

    #with open(out_rttm_dir+'/'+file_name+'.time', 'w') as fp:
    #  fp.write("%f %f %f %f %f\n" % (np.sum(ends-starts), ends[-1],  hac_time, vbx_time, vbf_time))
