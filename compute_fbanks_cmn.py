#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Authors: Lukas Burget, Federico Landini
# @Emails: burget@fit.vutbr.cz, landini@fit.vutbr.cz

import numpy as np
import soundfile as sf
import features
import struct
import kaldi_io
import diarization_lib
import sys

in_file_list = sys.argv[1]
in_lab_dir = sys.argv[2] # directory with files with VAD information
in_flac_dir = sys.argv[3] # directory with flac files
out_ark_fn = sys.argv[4] # ark file with cmn FBANK features (one record per speech segment)
out_seg_fn = sys.argv[5] # seg file further segmenting speech segment to subsegments (0.25 s shift, 1.5 s window)

file_names = np.loadtxt(in_file_list, dtype=object)

noverlap = 240
winlen = 400
fs = 16000
window = features.povey_window(winlen)
fbank_mx = features.mel_fbank_mx(winlen, fs, NUMCHANS=40, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
LC = 150
RC = 149

with open(out_seg_fn, "w") as seg_file:
  with open(out_ark_fn, "wb") as ark_file:
    for fn in file_names:
      labs = (np.loadtxt(in_lab_dir+"/"+fn+".lab", usecols=(0,1))*16000).astype(int)
      signal, samplerate = sf.read(in_flac_dir+"/"+fn+".flac")
      signal = features.add_dither((signal*2**(samplerate/1000 - 1)).astype(int))
      for segnum in range(len(labs)):
        seg=signal[labs[segnum,0]:labs[segnum,1]]
        seg=np.r_[seg[noverlap//2-1::-1], seg, seg[-1:-winlen//2-1:-1]] # Mirror noverlap//2 initial and final samples
        fea = features.fbank_htk(seg, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
        fea = features.cmvn_floating_kaldi(fea, LC, RC, norm_vars=False)      
        slen = len(fea)
        start=-25
        for start in range(0,slen-150,25):
          seg_file.write("%s_%04d-%08d-%08d %s %g %g\n" % (fn, segnum, start, (start+150), fn, labs[segnum,0]/float(fs)+start/100., labs[segnum,0]/float(fs)+start/100.+1.5))
          kaldi_io.write_mat(ark_file, fea[start:start+150], key="%s_%04d-%08d-%08d" % (fn, segnum, start, (start+150)))
        if slen-start-25 > 10:
          seg_file.write("%s_%04d-%08d-%08d %s %g %g\n" % (fn, segnum, start+25, slen, fn, labs[segnum,0]/float(fs)+(start+25)/100., labs[segnum,1]/float(fs)))
          kaldi_io.write_mat(ark_file, fea[start+25:slen], key="%s_%04d-%08d-%08d" % (fn, segnum, start+25, slen))
