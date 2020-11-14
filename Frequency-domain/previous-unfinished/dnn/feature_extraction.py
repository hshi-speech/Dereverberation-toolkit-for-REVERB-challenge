"""
Summary:  Feature Extraction. 
Author:   Hao Shi
Created:  2020.06.02
Modified: - 
"""

import argparse
import os,sys
sys.path.append('.')

import multiprocessing
from io_funcs.signal_processing import audiowrite, stft, audioread
import tensorflow as tf
import numpy as np
import types
import math
import random
import string
import pickle
import json


def mkdir_p(path):
    """ Creates a path recursively without throwing an error if it already exists

    :param path: path to create
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)

def read_files(path):
    """ Read files in ./lists

    :param path: path to read
    :return: lines
    """
    fid = open(path, 'r')
    lines = fid.readlines()
    fid.close()
    return lines

def concat_same(context, number):
    """ concatenate same frames (for the 1st or end frame)

    :param context: frames need to be concatenated
    :param number: number of concatenation
    :return: lines
    """
    buffer = context
    for i in range(0, number - 1):
       buffer = np.concatenate((buffer, context), axis=0) 
    return buffer

def concat_inputs(context, num_frames, adjacent_frames):
    """ concatenate frames for input

    :param context: frames need to be concatenated
    :param num_frames: number of frames
    :return: lines
    """
    buffer = context[0:num_frames, :]
    for i in range(0, adjacent_frames*2):
       buffer = np.concatenate((buffer, context[i + 1 : num_frames + i + 1, :]), axis=1) 
    return buffer

def gen_feats(wavs_clean, wavs_reverb, files_reverb):
    wav_name_clean, wav_path_clean = wavs_clean.split()
    wav_name_reverb, wav_path_reverb = wavs_reverb.split()

    if(wav_name_clean == wav_name_reverb):   
        mix_wav = audioread(wav_path_reverb, offset=0.0, duration=None, sample_rate=sample_rate)
        cln_wav = audioread(wav_path_clean, offset=0.0, duration=None, sample_rate=sample_rate)

        """
        Because of the length of reverberate wavs are larger than clean ones, 
        the mix_wav should be chosen the same number as clean one in front.
        """
        if(len(cln_wav)<len(mix_wav)):         
            mix_wav = mix_wav[0:len(cln_wav)]

        mix_stft = stft(mix_wav, time_dim=0, size=stft_size, shift=stft_shift)
        cln_stft = stft(cln_wav, time_dim=0, size=stft_size, shift=stft_shift)

        mix_abs = np.abs(mix_stft)
        mix_angle = np.angle(mix_stft)
        cln_abs = np.abs(cln_stft)

        [m, n] = mix_abs.shape
        adjacent_frames = math.ceil((num_concat - 1) / 2)
        pre_concat = concat_same(mix_abs[0, :].reshape((1,-1)), adjacent_frames)
        end_concat = concat_same(mix_abs[n - 1, :].reshape((1,-1)), adjacent_frames)
        utterance = np.concatenate((pre_concat, mix_abs, end_concat), axis = 0)
        inputs_utterance = concat_inputs(utterance, m, adjacent_frames)
        
        sums = np.sum(mix_abs, axis=0)

        if(shuffles):
            for i in range (0, m):
                ran_name = ''.join(random.sample(string.ascii_letters + string.digits, 20))
                out_feat_path = './workspace/dump/' + os.path.splitext(files_reverb)[0] + '/' + ran_name + '.p'
                print(datasets  + '    ' + out_feat_path)
                data = [inputs_utterance[i, :], cln_abs[i, :], mix_angle[i, :]]
                pickle.dump(data, open(out_feat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            out_feat_path = './workspace/dump/' + os.path.splitext(files_reverb)[0] + '/' + wav_name_reverb + '.p'
            print(datasets  + '   ' + out_feat_path)
            data = [inputs_utterance, cln_abs, mix_angle]
            pickle.dump(data, open(out_feat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        return sums, m

def dump_feats():
    lines_clean = read_files('./lists/' + datasets + '_clean.lst')
    lines_reverb =  read_files('./lists/' + datasets + '_reverb.lst')

    for files_clean, files_reverb in zip(lines_clean, lines_reverb):
      mkdir_p('./workspace/dump/' + os.path.splitext(files_reverb)[0] + '/')
      mkdir_p('./workspace/dump/scaler/')

      files_clean = files_clean.strip('\n')
      files_reverb = files_reverb.strip('\n')

      wavLines_clean = read_files(files_clean)
      wavLines_reverb = read_files(files_reverb)

      buffer = np.array([])
      num_frames = 0

      for wavs_clean, wavs_reverb in zip(wavLines_clean, wavLines_reverb):
        sums, m = gen_feats(wavs_clean, wavs_reverb, files_reverb)
        if(datasets == 'tr'):
            if(buffer.size == 0):
                buffer = sums
            else:
                buffer += sums

            num_frames += m

      if(datasets == 'tr'):
        out_feat_path = './workspace/dump/scaler/scaler.p'
        data = [buffer, num_frames]
        pickle.dump(data, open(out_feat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print('Features Dumping Is Done!')

def datas():
    file_infos = []
    lines_clean = read_files('./lists/' + datasets + '_clean.lst')
    lines_reverb =  read_files('./lists/' + datasets + '_reverb.lst')

    for files_clean, files_reverb in zip(lines_clean, lines_reverb):
        out_dir = './workspace/dump/datas/'
        mkdir_p(out_dir)
        files_reverb = files_reverb.strip('\n')
        files_clean = files_clean.strip('\n')

        in_dir = os.path.abspath('./workspace/dump/' + os.path.splitext(files_reverb)[0])
        wav_list = os.listdir(in_dir)
        for wav_file in wav_list:
            if not wav_file.endswith('.p'):
                continue
            wav_path = os.path.join(in_dir, wav_file)
            file_infos.append((wav_path))

        with open(os.path.join(out_dir, datasets + '.json'), 'w') as f:
            json.dump(file_infos, f, indent=4)

    print('Output The Files in The Folder Is Done!')

def variance_cal():
    scalers = pickle.load(open('./workspace/dump/scaler/scaler.p', 'rb'))
    [means, num_frames] = scalers 

    with open('./workspace/dump/datas/tr.json', 'r') as f:
        tr_infos = json.load(f)

    def sort(infos): return sorted(
            infos, key=lambda info: info[0], reverse=True)

    buffer = np.array([])
    sorted_tr_infos = sort(tr_infos)
    for paths in sorted_tr_infos:
        data = pickle.load(open(paths, 'rb'))
        [inputs, cln_abs, mix_angle] = data
        inputs = inputs.reshape((num_concat, -1))
        squares = np.square(inputs[math.ceil((num_concat - 1) / 2), :] - means / num_frames)

        if(buffer.size == 0):
            buffer = squares / num_frames
        else:
            buffer += squares / num_frames

    out_feat_path = './workspace/dump/scaler/scalers.p'
    data = [means / num_frames, buffer, num_frames]
    pickle.dump(data, open(out_feat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print('Variance Computing Is Done!')


###
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Datasets for Pytorch-based Speech Dereverberation')
    parser.add_argument('num_concat', help='concatenation number of frames')
    parser.add_argument('normalization', help='whether to normalize the features')
    parser.add_argument('datasets', help='which datasets do you want to create')
    parser.add_argument('shuffles', help='whether the features are randomly numbered')
    parser.add_argument('stft_size', help='size of STFT')
    parser.add_argument('stft_shift', help='shift of STFT')
    parser.add_argument('sample_rate', help='sample rate of waveform')

    args = parser.parse_args()
    num_concat = int(args.num_concat)
    normalization = int(args.normalization)
    datasets = args.datasets
    shuffles = int(args.shuffles)
    stft_size = int(args.stft_size)
    stft_shift = int(args.stft_shift)
    sample_rate = int(args.sample_rate)


    ### First: dump all features into folders ###
    # dump_feats()

    ### Second: output the files in the folder ###
    # datas()

    ### Third: calculate the variance ###
    variance_cal()



