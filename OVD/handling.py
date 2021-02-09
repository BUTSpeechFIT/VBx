#!/usr/bin/env python

# @Authors: Federico Landini, Mireia Diez
# @Emails: landini@fit.vutbr.cz, mireia@fit.vutbr.cz
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


import sys, os, errno
import numpy as np
import operator
import subprocess
import copy


def add_speakers(spks1, spks2):
	res = copy.deepcopy(spks1)
	new_speakers = get_pos_not_defined(spks1, spks2)
	if len(new_speakers) > 0:
		res[new_speakers[0]] = 1
	return res


def get_closest_speaker_left(m, pos):
	speakers = copy.deepcopy(m[pos])
	qty_speakers = np.sum(speakers)
	dist = 1
	while qty_speakers<2 and pos-dist > 0:
		new_speakers = get_pos_not_defined(m[pos], m[pos-dist])
		if len(new_speakers) > 0:
			for s in new_speakers:
				speakers[s] = 1
				qty_speakers = np.sum(speakers)
		dist += 1
	return speakers, dist


def get_closest_speaker_right(m, pos):
	speakers = copy.deepcopy(m[pos])
	qty_speakers = np.sum(speakers)
	dist = 1
	while pos+dist < m.shape[0]:
		new_speakers = get_pos_not_defined(m[pos], m[pos+dist])
		if len(new_speakers) > 0:
			for s in new_speakers:
				speakers[s] = 1
				qty_speakers = np.sum(speakers)
		dist += 1
	return speakers, dist


def get_pos_not_defined(spks1, spks2):
	# Find those speakers defined in spks1 but not in spks2
	speakers = []
	for i in range(spks1.shape[0]):
		if not(spks1[i]) and spks2[i]:
			speakers.append(i)
	return speakers


def hard_labels_to_rttm(labels, id_file, out_rttm_file, frameshift=0.01):
	""" hard_labels_to_rttm(labels, id_file, out_rttm_file, frameshift)
	  takes a NfxNs matrix encoding the frames in which each speaker is present (labels 1/0) where
	  Ns is the number of speakers and Nf is the resulting number of frames, according to the parameters given
	  Nf might be shorter than the real number of frames of the utterance, as final silence parts cannot be recovered from the rttm
	  In case of silence all speakers are labeled with 0,
	  In case of overlap all speakers involved are marked with 1
	  The function assumes that the rttm only contains speaker turns (no silences), the overlaps are extracted from the speaker turn collisions
	  It generates an rttm file where the speaker labels are numbers in the range 0:Ns
	  It uses the frameshift to define the resolution used for reading the matrix
	"""
	if len(labels.shape) > 1:
		# Delete columns with all 0's: excess of speakers
		non_empty_spks = np.where(labels.sum(axis=0)!=0)[0]
		labels = labels[:,non_empty_spks]

	# Add initial 0's: to mark the initial frame and use diff
	if len(labels.shape) > 1:
		labels = np.vstack([np.zeros((1,labels.shape[1])), labels])
	else:
		labels = np.insert(labels, 0, 0)
	d = np.diff(labels, axis=0)
	f = open(out_rttm_file, 'w')

	spk_list = []
	ini_list = []
	end_list = []

	if len(labels.shape) > 1:
		n_spks = labels.shape[1]
	else:
		n_spks = 1
	for spk in range(n_spks):
		if n_spks > 1:
			ini_indices = np.where(d[:,spk]==1)[0] -1
			end_indices = np.where(d[:,spk]==-1)[0] -1
		else:
			ini_indices = np.where(d[:]==1)[0] -1
			end_indices = np.where(d[:]==-1)[0] -1
		# Add final mark if needed
		if (len(ini_indices) == len(end_indices) + 1):
			end_indices = np.hstack([end_indices, labels.shape[0]])
		assert(len(ini_indices)==len(end_indices))
		n_segments = len(ini_indices)
		for index in range(n_segments):
			spk_list.append(spk)
			ini_list.append(ini_indices[index])
			end_list.append(end_indices[index])

	for ini,end,spk in sorted(zip(ini_list, end_list, spk_list)):
		f.write('SPEAKER ' + id_file + ' 1 ' + str(round(ini*frameshift+frameshift, 3)) + ' ' + str(round((end-ini)*frameshift, 3)) + ' <NA> <NA> ' + 'spk' + str(spk) + ' <NA> <NA>\n')
	f.close()


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def rttm_to_hard_labels(rttm_path, framerate=25, frameshift=10, length=None, n_spks=None):
	""" rttm_to_hard_labels(rttm_path, framerate=25, frameshift=10, length=None, n_spks=None)
		reads the rttm and returns a NfxNs matrix encoding the frames in which each speaker is present (labels 1/0)
		Ns is the number of speakers and Nf is the resulting number of frames, according to the parameters given
		Nf might be shorter than the real number of frames of the utterance, as final silence parts cannot be recovered from the rttm
		The optional parameters allow for more frames or speakers.
		In case of silence all speakers are labeled with 0,
		In case of overlap all speakers involved are marked with 1
		The function assumes that the rttm only contains speaker turns (no silences), the overlaps are extracted from the speaker turn collisions
	"""

	data  = np.loadtxt(rttm_path, usecols=[3,4]) # each row is a turn, columns denote beggining of the turn (s) and duration (s)
	spks  = np.loadtxt(rttm_path, usecols=[7],dtype='str') # spk id of each turn
	if spks.ravel().shape[0] == 1:
		spks = np.asarray(spks.ravel())
		data = np.asarray([data])
	if spks.shape[0] > 0:
		spk_ids = np.unique(spks)
		Ns = len(spk_ids)
		if n_spks is not None:
			n_spks = max(Ns, n_spks)
		else:
			n_spks = Ns

		if length == None:
			len_file = data[-1][0]+data[-1][1] # this is the lenght of the file (s) that can be recovered from the rttm, there might be extra silence at the end
		else:
			len_file = length

		labels = np.zeros([int(len_file*1000),n_spks]) # vector of labels in ms precision
		ranges = (np.round_(np.array([data[:,0],data[:,0]+data[:,1]]).T*1000)).astype(int) # ranges to mark each turn

		for s in range(Ns): # we loop over speakers
			for init_end in ranges[spks==spk_ids[s],:]: # loop in all the turns of the speaker
				labels[init_end[0]:init_end[1],s]=1     # mark the spk

		fr_labels = labels[framerate//2::frameshift,:] # downsampling the input to the frame rate 

		return np.asarray(fr_labels)
	else:
		return np.asarray([])


def main():
	import argparse
	parser = argparse.ArgumentParser(description='Handle overlapped speech')
	parser.add_argument('--in-rttm-dir', required=True, type=str, help='dir with rttm files on which to add overlap')
	parser.add_argument('--in-2nd-rttm-dir', required=False, type=str, help='dir with second rttm files from which to take speaker labels')
	parser.add_argument('--in-rttm-with-ov-dir', required=True, type=str, help='dir with rttm files that have overlap segments')
	parser.add_argument('--out-rttm-dir', required=True, type=str, help='out directory where overlap segments will be assigned to speakers')
	approach = parser.add_mutually_exclusive_group(required=True)
	approach.add_argument('--heuristic', type=bool)
	approach.add_argument('--label2nd', type=bool)
	parser.add_argument('--in-file-list', required=True, help='txt file that lists the utterances to process')
	args = parser.parse_args()

	if args.heuristic:
		files_list = np.loadtxt(args.in_file_list, dtype='object')
		if files_list.shape == (1,1):
			files_list = files_list[0]
		for key in files_list:
			print(key)
			matrix_rttm = rttm_to_hard_labels("%s/%s.rttm" % (args.in_rttm_dir, key), framerate=1, frameshift=1)
			if matrix_rttm.shape[1] > 1 and os.stat("%s/%s.rttm" % (args.in_rttm_with_ov_dir, key)).st_size > 0:
				matrix_ov = rttm_to_hard_labels("%s/%s.rttm" % (args.in_rttm_with_ov_dir, key), framerate=1, frameshift=1)
				# Extend both matrices with the maximum length
				max_length = max(matrix_rttm.shape[0], matrix_ov.shape[0])
				if max_length > matrix_ov.shape[0]:
					matrix_ov = np.concatenate((matrix_ov, np.zeros((max_length - matrix_ov.shape[0], matrix_ov.shape[1]))))
				if max_length > matrix_rttm.shape[0]:
					matrix_rttm = np.concatenate((matrix_rttm, np.zeros((max_length - matrix_rttm.shape[0], matrix_rttm.shape[1]))))
				out_matrix = []
				for i in range(matrix_rttm.shape[0]):
					if matrix_ov[i]:
						if i>0 and matrix_ov[i-1]:
							dist_left += 1
							dist_right -= 1
						else:
							closest_left, dist_left = get_closest_speaker_left(matrix_rttm, i)
							closest_right, dist_right = get_closest_speaker_right(matrix_rttm, i)
						current_and_closest_left = add_speakers(matrix_rttm[i], closest_left)
						current_and_closest_right = add_speakers(matrix_rttm[i], closest_right)
						if (np.sum(current_and_closest_left) > 1 and
							dist_left < dist_right or np.sum(current_and_closest_right) <= 1 and
							np.sum(current_and_closest_left) > 1):
								out_matrix.append(current_and_closest_left)
						elif np.sum(current_and_closest_right) > 1:
							out_matrix.append(current_and_closest_right)
						else:
							out_matrix.append(matrix_rttm[i])
					else:
						out_matrix.append(matrix_rttm[i])
				out_matrix = np.asarray(out_matrix)
			else:
				out_matrix = matrix_rttm
			mkdir_p("%s/%s" % (args.out_rttm_dir, os.path.dirname(key)))
			hard_labels_to_rttm(out_matrix, key.split('/')[-1], "%s/%s.rttm" % (args.out_rttm_dir, key), frameshift=0.001)

			bashCommand = "sort -k 4 -n -o %s/%s.rttm %s/%s.rttm" % (args.out_rttm_dir, key, args.out_rttm_dir, key)
			process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
			output, error = process.communicate()
	
	elif args.label2nd:
		if args.in_2nd_rttm_dir is None:
			print("--in-2nd-rttm-dir has to be defined to use label2nd.")
		else:
			files_list = np.loadtxt(args.in_file_list, dtype='object')
			if files_list.shape == (1,1):
				files_list = files_list[0]
			for key in files_list:
				print(key)
				in_matrix_1st = rttm_to_hard_labels("%s/%s.rttm" % (args.in_rttm_dir, key), framerate=1, frameshift=1)
				in_matrix_2nd = rttm_to_hard_labels("%s/%s.rttm" % (args.in_2nd_rttm_dir, key), n_spks=in_matrix_1st.shape[1], framerate=1, frameshift=1)
				length = max(in_matrix_1st.shape[0], in_matrix_2nd.shape[0])/1000.0 # shapes are in ms
				in_matrix_1st = rttm_to_hard_labels("%s/%s.rttm" % (args.in_rttm_dir, key), length=length, framerate=1, frameshift=1)
				in_matrix_2nd = rttm_to_hard_labels("%s/%s.rttm" % (args.in_2nd_rttm_dir, key), length=length, n_spks=in_matrix_1st.shape[1], framerate=1, frameshift=1)
				if in_matrix_1st.shape[1] > 1 and os.stat("%s/%s.rttm" % (args.in_rttm_with_ov_dir, key)).st_size > 0:
					in_ov = rttm_to_hard_labels("%s/%s.rttm" % (args.in_rttm_with_ov_dir, key), length=length, framerate=1, frameshift=1)
					max_length = max(in_matrix_1st.shape[0], in_ov.shape[0])
					if max_length > in_ov.shape[0]:
						in_ov = np.concatenate((in_ov, np.zeros(max_length - in_ov.shape[0])))
					if max_length > in_matrix_1st.shape[0]:
						in_matrix_1st = np.concatenate((in_matrix_1st, np.zeros((max_length - in_matrix_1st.shape[0], in_matrix_1st.shape[1]))))
						in_matrix_2nd = np.concatenate((in_matrix_2nd, np.zeros((max_length - in_matrix_2nd.shape[0], in_matrix_2nd.shape[1]))))
					out_matrix = []

					in_matrix_2nd = in_matrix_2nd[:in_matrix_1st.shape[0],:]
					in_ov = in_ov[:in_matrix_1st.shape[0]]
					if in_matrix_1st.shape[1] != in_matrix_2nd.shape[1]:
						print("different speakers "+key+" keeping one")
					else:
						if in_matrix_1st.shape[1] > 1:
							ov_pos = np.where(in_ov == 1)[0]
							in_matrix_1st[ov_pos] += in_matrix_2nd[ov_pos]

				# If for some reason some speaker has overlap with themselves, remove the duplicate
				in_matrix_1st[np.where(in_matrix_1st > 1)] = 1

				mkdir_p("%s/%s" % (args.out_rttm_dir, os.path.dirname(key)))
				hard_labels_to_rttm(in_matrix_1st, key, "%s/%s.rttm" % (args.out_rttm_dir, key), frameshift=0.001)
	
	else:
		print("One of the options has to be selected: heuristic or 2ndlabel")


if __name__ == "__main__":
	# execute only if run as a script
	main()
