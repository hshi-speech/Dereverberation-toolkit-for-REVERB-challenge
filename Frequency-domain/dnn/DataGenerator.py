"""
Summary:  Data Generator. 
Author:   Hao Shi
Created:  2020.06.02
Modified: - 
"""

import json
import math
import os

import numpy as np
import torch
import torch.utils.data as data
import pickle

class FramesDataset(data.Dataset):

    def __init__(self, datasets, batch_size):
        super(FramesDataset, self).__init__()
        with open('/workspace/dump/datas/' + datasets + '.json', 'r') as f:
            infos = json.load(f)
        def sort(infos): return sorted(
            infos, key=lambda info: int(info[0]), reverse=True)
        sorted_infos = sort(infos)

        minibatch = []
        start = 0
        while True:
            num_segments = 0
            end = start
            part_mix = []

            while num_segments < batch_size and end < len(sorted_infos):
                part_mix.append(sorted_infos[end])
                end += 1

            if len(part_mix) > 0:
                minibatch.append([part_mix, end-start])

            if end == len(sorted_infos):
                break
            start = end

        self.minibatch = minibatch


    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class FramesDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(FramesDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


def load_mixtures_and_sources(batch):
    mixtures, sources, phases = [], [], []
    mix_infos, minibatch_size = batch

    for i in range(0, minibatch_size):
        data = pickle.load(open(mix_infos[i], 'rb'))
        [inputs, cln_abs, mix_angle] = data
        
        if(mixtures.size == 0):
            mixtures = inputs
            sources = cln_abs
            phases = mix_angle
        else:
            mixtures = np.concatenate((mixtures, inputs), axis=1)
            sources = np.concatenate((sources, cln_abs), axis=1)
            phases = np.concatenate((phases, mix_angle), axis=1)

    return mixtures, sources, phases
        

def _collate_fn(batch):
    mixtures, sources, phases = load_mixtures_and_sources(batch)
    return torch.from_numpy(mixtures), torch.from_numpy(sources), torch.from_numpy(phases)













