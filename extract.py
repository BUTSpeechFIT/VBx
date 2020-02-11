#!/usr/bin/env python

# -*- coding: utf-8 -*-
# @Authors: Shuai Wang, Federico Landini
# @Emails: wsstriving@gmail.com, landini@fit.vutbr.cz

import torch
import sys, os
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import numpy as np
import kaldi_io
from tdnn_model import load_kaldi_model

torch.backends.cudnn.benchmark=True


def validate_path(dir_name):
    """
    :param dir_name: Create the directory if it doesn't exist
    :return: None
    """
    dir_name = os.path.dirname(dir_name)  # get the path
    if not os.path.exists(dir_name) and (dir_name is not ''):
        os.makedirs(dir_name)


class KaldiDatasetUtt_ARK(Dataset):
    """
    Utilize the generator properties
    WARNING: The data loader should set num_workers to 1!
    """
    def __init__(self, feat_ark, feats_len, min_len=25):
        super(KaldiDatasetUtt_ARK, self).__init__()
        self.data_generator = kaldi_io.read_mat_ark(feat_ark)
        self.min_len = min_len
        self.length = feats_len

    def __getitem__(self, idx):
        name, feat = next(self.data_generator)
        if len(feat) < self.min_len:
            left_pad = ((self.min_len) - len(feat)) // 2
            right_pad = self.min_len - len(feat) - left_pad 
            feat = np.pad(feat, ((left_pad, right_pad),(0,0)), 'edge')
        return feat, name

    def __len__(self):
        return self.length


def write_ark(model, args):
    """
    Write the extracted embeddings to ark file, having the same format as i-vectors
    """
    dataset = KaldiDatasetUtt_ARK(args.feats_ark, args.feats_len)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    model.eval()
    save_to = args.ark_file
    validate_path(save_to)

    with torch.no_grad():
        with open(save_to, 'wb') as save_to_file:
            for batch_idx, (data, names) in tqdm(enumerate(data_loader), ncols=100, total=len(data_loader)):
                data = data.transpose(1,2)
                data = data.to(args.cuda, dtype=torch.double)
                _, embedding_a, embedding_b = model(data)
                vectors = embedding_a.data.cpu().numpy()
                for i in range(len(vectors)):
                    kaldi_io.write_vec_flt(save_to_file, vectors[i], names[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats-ark', type=str, help="Input feats ark")
    parser.add_argument('--feats-len', type=int, help="Number of feats")
    parser.add_argument('--ark-file', type=str, help="Output embeddings ark")
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument("--model-init", type=str, default=None)

    args = parser.parse_args()

    if args.model_init is not None:
        model = load_kaldi_model(args.model_init)
    print(model)
    
    args.cuda = torch.device("cpu")
    #args.cuda = torch.device("cuda")
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    model = model.to(args.cuda, dtype=torch.double)
    write_ark(model, args)


if __name__ == '__main__':
    main()
