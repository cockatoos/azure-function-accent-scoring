import pandas as pd       
import os 
import math 
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
import torch
import torch.nn as nn
import pickle
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

def load_model(modelPath):
    """ Load accent classification model from modelPath."""
    # modelPath = '../new_model.pt'
    model = torch.load(modelPath)
    return model

def generate_mfcc_data(mfcc):
    mfcc_standardized = np.zeros(mfcc.shape)
    for b in range(mfcc.shape[0]):
        mfcc_slice = mfcc[b,:]
        centered = mfcc_slice - np.mean(mfcc_slice)
        if np.std(centered) != 0:
            centered_scaled = centered / np.std(centered)

        mfcc_standardized[b,:] = centered_scaled

    delta1 = librosa.feature.delta(mfcc_standardized, order=1)
    delta2 = librosa.feature.delta(mfcc_standardized, order=2)
    mfcc_data = np.stack((mfcc_standardized,delta1,delta2))

    return mfcc_data

def segment_and_standardize_audio(path, seg_thresh):
    sound_file = AudioSegment.from_mp3(path)
    audio_chunks = split_on_silence(sound_file, 
        # must be silent for at least half a second
        min_silence_len = 80,
        # consider it silent if quieter than -16 dBFS
        silence_thresh=-30
    )
    standardized_chunks = []

    for seg in audio_chunks:
        seg_len = len(seg)
        if seg_len >= seg_thresh:
            seg_standardized = seg[0:seg_thresh]
        else:
            seg_standardized = seg + AudioSegment.silent(duration=(seg_thresh - seg_len))
        standardized_chunks.append(seg_standardized)

    return standardized_chunks

def classify_accent(test_dir, model_path):
    """
    Classify audio samples in given directory.

    Args:
        test_dir (str): directory path containing audio samples
        model_path (str): path that stores PyTorch model

    Returns:
        dict (str: int): Mapping of audio filenames to classification result
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path)
    predictions = dict()
    for f in os.listdir(test_dir):
        audio_chunks = segment_and_standardize_audio(test_dir + '/' + f, 500)
        num_english_pred = 0
        for seg in audio_chunks:

            samples = seg.get_array_of_samples()
            arr = np.array(samples).astype(np.float32)/32768 # 16 bit 
            arr = librosa.core.resample(arr, seg.frame_rate, 22050, res_type='kaiser_best') 

            mfcc = librosa.feature.mfcc(y=arr, sr=22050)
            data = generate_mfcc_data(mfcc)
            pred = model(torch.from_numpy(data).unsqueeze(0).float().to(device)).item()
            if pred > 0.5:
                num_english_pred += 1
        
        frac_english_preds = num_english_pred / len(audio_chunks)
        
        if frac_english_preds >= 0.5:
            predictions[f] = 1
        else:
            predictions[f] = 0

    # there should only be one item in the predictions
    return random.choice(list(predictions.values()))  