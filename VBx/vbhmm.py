#!/usr/bin/env python

# @Authors: Lukas Burget, Mireia Diez, Federico Landini, Jan Profant
# @Emails: burget@fit.vutbr.cz, mireia@fit.vutbr.cz, landini@fit.vutbr.cz, jan.profant@phonexia.com
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


# The recipe consists in doing Agglomerative Hierachical Clustering on
# x-vectors in a first step. Then, Variational Bayes HMM over x-vectors
# is applied using the AHC output as args.initialization.
#
# A detailed analysis of this approach is presented in
# F. Landini, J. Profant, M. Diez, L. Burget
# Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization:
# theory, implementation and analysis on standard tasks
# Computer Speech & Language, 2022

import argparse
import os
import itertools

import fastcluster
import h5py
import kaldi_io
import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.special import softmax
from scipy.linalg import eigh

from diarization_lib import read_xvector_timing_dict, l2_norm, \
    cos_similarity, twoGMMcalib_lin, merge_adjacent_labels, mkdir_p
from kaldi_utils import read_plda
from VBx import VBx


def write_output(fp, out_labels, starts, ends):
    for label, seg_start, seg_end in zip(out_labels, starts, ends):
        fp.write(f'SPEAKER {file_name} 1 {seg_start:03f} {seg_end - seg_start:03f} '
                 f'<NA> <NA> {label + 1} <NA> <NA>{os.linesep}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', required=True, type=str,
                        choices=['AHC', 'AHC+VB'],
                        help='AHC for using only AHC or AHC+VB for VB-HMM '
                        'after AHC initilization', )
    parser.add_argument('--out-rttm-dir', required=True, type=str,
                        help='Directory to store output rttm files')
    parser.add_argument('--xvec-ark-file', required=True, type=str,
                        help='Kaldi ark file with x-vectors from one or more '
                             'input recordings. Attention: all x-vectors from '
                             'one recording must be in one ark file')
    parser.add_argument('--segments-file', required=True, type=str,
                        help='File with x-vector timing info. See '
                        'diarization_lib.read_xvector_timing_dict')
    parser.add_argument('--xvec-transform', required=True, type=str,
                        help='path to x-vector transformation h5 file')
    parser.add_argument('--plda-file', required=True, type=str,
                        help='File with PLDA model in Kaldi format used for '
                        'AHC and VB-HMM x-vector clustering')
    parser.add_argument('--threshold', required=True, type=float,
                        help='args.threshold (bias) used for AHC')
    parser.add_argument('--lda-dim', required=True, type=int,
                        help='For VB-HMM, x-vectors are reduced to this '
                        'dimensionality using LDA')
    parser.add_argument('--Fa', required=True, type=float,
                        help='Parameter of VB-HMM (see VBx.VBx)')
    parser.add_argument('--Fb', required=True, type=float,
                        help='Parameter of VB-HMM (see VBx.VBx)')
    parser.add_argument('--loopP', required=True, type=float,
                        help='Parameter of VB-HMM (see VBx.VBx)')
    parser.add_argument('--target-energy', required=False, type=float,
                        default=1.0,
                        help='Parameter affecting AHC if the similarity '
                             'matrix is obtained with PLDA. See '
                             'diarization_lib.kaldi_ivector_plda_scoring_dense')
    parser.add_argument('--init-smoothing', required=False, type=float,
                        default=5.0,
                        help='AHC produces hard assignments of x-vetors to '
                             'speakers. These are "smoothed" to soft '
                             'assignments as the args.initialization for '
                             'VB-HMM. This parameter controls the amount of '
                             'smoothing. Not so important, high value '
                             '(e.g. 10) is OK  => keeping hard assigment')
    parser.add_argument('--output-2nd', required=False, type=bool, default=False,
                        help='Output also second most likely speaker of VB-HMM')

    args = parser.parse_args()
    assert 0 <= args.loopP <= 1, f'Expecting loopP between 0 and 1, got {args.loopP} instead.'

    # segments file with x-vector timing information
    segs_dict = read_xvector_timing_dict(args.segments_file)

    kaldi_plda = read_plda(args.plda_file)
    plda_mu, plda_tr, plda_psi = kaldi_plda
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]

    # Open ark file with x-vectors and in each iteration of the following
    # for-loop read a batch of x-vectors corresponding to one recording
    arkit = kaldi_io.read_vec_flt_ark(args.xvec_ark_file)
    # group xvectors in ark by recording name
    recit = itertools.groupby(arkit, lambda e: e[0].rsplit('_', 1)[0])
    for file_name, segs in recit:
        print(file_name)
        seg_names, xvecs = zip(*segs)
        x = np.array(xvecs)

        with h5py.File(args.xvec_transform, 'r') as f:
            mean1 = np.array(f['mean1'])
            mean2 = np.array(f['mean2'])
            lda = np.array(f['lda'])
            x = l2_norm(lda.T.dot((l2_norm(x - mean1)).transpose()).transpose() - mean2)

        if args.init == 'AHC' or args.init.endswith('VB'):
            if args.init.startswith('AHC'):
                # Kaldi-like AHC of x-vectors (scr_mx is matrix of pairwise
                # similarities between all x-vectors)
                scr_mx = cos_similarity(x)
                # Figure out utterance specific args.threshold for AHC
                thr, _ = twoGMMcalib_lin(scr_mx.ravel())
                # output "labels" is an integer vector of speaker (cluster) ids
                scr_mx = squareform(-scr_mx, checks=False)
                lin_mat = fastcluster.linkage(
                    scr_mx, method='average', preserve_input='False')
                del scr_mx
                adjust = abs(lin_mat[:, 2].min())
                lin_mat[:, 2] += adjust
                labels1st = fcluster(lin_mat, -(thr + args.threshold) + adjust,
                    criterion='distance') - 1
            if args.init.endswith('VB'):
                # Smooth the hard labels obtained from AHC to soft assignments
                # of x-vectors to speakers
                qinit = np.zeros((len(labels1st), np.max(labels1st) + 1))
                qinit[range(len(labels1st)), labels1st] = 1.0
                qinit = softmax(qinit * args.init_smoothing, axis=1)
                fea = (x - plda_mu).dot(plda_tr.T)[:, :args.lda_dim]
                q, sp, L = VBx(
                    fea, plda_psi[:args.lda_dim],
                    pi=qinit.shape[1], gamma=qinit,
                    maxIters=40, epsilon=1e-6,
                    loopProb=args.loopP, Fa=args.Fa, Fb=args.Fb)

                labels1st = np.argsort(-q, axis=1)[:, 0]
                if q.shape[1] > 1:
                    labels2nd = np.argsort(-q, axis=1)[:, 1]
        else:
            raise ValueError('Wrong option for args.initialization.')

        assert(np.all(segs_dict[file_name][0] == np.array(seg_names)))
        start, end = segs_dict[file_name][1].T

        starts, ends, out_labels = merge_adjacent_labels(start, end, labels1st)
        mkdir_p(args.out_rttm_dir)
        with open(os.path.join(args.out_rttm_dir, f'{file_name}.rttm'), 'w') as fp:
            write_output(fp, out_labels, starts, ends)

        if args.output_2nd and args.init.endswith('VB') and q.shape[1] > 1:
            starts, ends, out_labels2 = merge_adjacent_labels(start, end, labels2nd)
            output_rttm_dir = f'{args.out_rttm_dir}2nd'
            mkdir_p(output_rttm_dir)
            with open(os.path.join(output_rttm_dir, f'{file_name}.rttm'), 'w') as fp:
                write_output(fp, out_labels2, starts, ends)
