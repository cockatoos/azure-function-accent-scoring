import sys
sys.path.append('/usr/bin/ffmpeg')

import math
import os
import pickle
import random
from pathlib import Path

import logging
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import torch
from pydub import AudioSegment
from pydub.silence import split_on_silence
import onnx
import onnxruntime


############################## PyTorch model is not used in Azure Function. ###########################
##############################      Uncomment for debugging purposes.       ###########################
# def load_model(modelPath):
#     """ Load accent classification model from modelPath."""
#     model = torch.load(modelPath)
#     return model

# def save_onnx_model(model, data, save_model_file):
#     model.eval()
#     torch.onnx.export(model, data, save_model_file, opset_version=11)

# def classify_accent_with_torch(model_path, data, save_onnx=False):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = load_model(model_path)
#     # save model in .onnx format to avoid importing torch in Azure Function
#     if save_onnx:
#         logging.info("Exporting .onnx model...")
#         save_onnx_model(model, torch.from_numpy(data).unsqueeze(0).float().to(device), "../eval_model.onnx")
#         logging.info(".onnx model saved.")

#     model.eval()
#     pred = model(torch.from_numpy(data).unsqueeze(0).float().to(device)).item()
#     print(f"result from pytorch model: {pred}")
#
####################################################################################################

def generate_mfcc_data(mfcc):
    mfcc_standardized = np.zeros(mfcc.shape)
    for b in range(mfcc.shape[0]):
        mfcc_slice = mfcc[b, :]
        centered = mfcc_slice - np.mean(mfcc_slice)
        if np.std(centered) != 0:
            centered_scaled = centered / np.std(centered)

        mfcc_standardized[b, :] = centered_scaled

    delta1 = librosa.feature.delta(mfcc_standardized, order=1)
    delta2 = librosa.feature.delta(mfcc_standardized, order=2)
    mfcc_data = np.stack((mfcc_standardized, delta1, delta2))

    return mfcc_data


def segment_and_standardize_audio(path, seg_thresh):
    sound_file = AudioSegment.from_mp3(path)
    audio_chunks = split_on_silence(
        sound_file,
        # must be silent for at least half a second
        min_silence_len=80,
        # consider it silent if quieter than -16 dBFS
        silence_thresh=-30,
    )
    standardized_chunks = []

    for seg in audio_chunks:
        seg_len = len(seg)
        if seg_len >= seg_thresh:
            seg_standardized = seg[0:seg_thresh]
        else:
            seg_standardized = seg + AudioSegment.silent(
                duration=(seg_thresh - seg_len)
            )
        standardized_chunks.append(seg_standardized)

    return standardized_chunks

# def segment_and_standardize_audio(path, seg_size):
#     sound_file = AudioSegment.from_mp3(path)
#     limit = len(sound_file) // seg_size if len(sound_file) % seg_size == 0 else len(sound_file) // seg_size + 1
#     chunks = []
#     for i in range(0,limit):
#         chunk = sound_file[i * seg_size : (i + 1) * seg_size]
#         if len(chunk) < seg_size:
#             chunk = chunk + AudioSegment.silent(duration=(seg_size - len(chunk)))
          

#         if np.count_nonzero(chunk.get_array_of_samples()) > 45000:
#             chunks.append(chunk)
#     return chunks


def classify_accent(test_dir, model_path, save_onnx=False):
    """
    Classify audio samples in given directory.

    Args:
        test_dir (str): directory path containing audio samples
        model_path (str): path that stores PyTorch model

    Returns:
        dict (str: int): Mapping of audio filenames to classification result
    """
    logging.info("Preparing for classification...")
    predictions = dict()
    for f in os.listdir(test_dir):
        logging.info("Segmenting audio...")
        logging.info(f"Filename: {f}")
        print(f"Filename: {f}")
        audio_chunks = segment_and_standardize_audio(test_dir + "/" + f, 1000)
        print(f"length of audio chunks: {len(audio_chunks)}")
        logging.info("Segmentation complete.")
        num_english_pred = 0
        prob_english_pred = 0
        for seg in audio_chunks:

            samples = seg.get_array_of_samples()
            arr = np.array(samples).astype(np.float32) / 32768  # 16 bit
            arr = librosa.core.resample(
                arr, seg.frame_rate, 22050, res_type="kaiser_best"
            )
            mfcc = librosa.feature.mfcc(y=arr, sr=22050, n_mfcc=50)
            logging.info("generating mfcc data...")
            data = generate_mfcc_data(mfcc)
            logging.info("mfcc data generated.")
            
            # Load .onnx model and verify correctness
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            # Predict with onnx model
            sess = onnxruntime.InferenceSession(model_path)
            input_name = sess.get_inputs()[0].name
            pred = sess.run(None, {input_name: np.expand_dims(data.astype(np.float32), axis=0)})[0]
            print(f"result from .onnx model: {pred}")
            logging.info(f"result from .onnx model: {pred}")

            # Uncomment this to compare results of .onnx and pytorch model.
            # classify_accent_with_torch(model_path, data, save_onnx=save_onnx)
            prob_english_pred += pred

            if pred > 0.5:
                num_english_pred += 1

        frac_english_preds = num_english_pred / len(audio_chunks)
        prob_english_preds = prob_english_pred / len(audio_chunks)

        # if frac_english_preds >= 0.5:
        #     predictions[f] = 1
        # else:
        #     predictions[f] = 0

    # there should only be one item in the predictions
    # score = random.choice(list(predictions.values()))
    print(f"prob_english_preds: {prob_english_preds}")
    return {"status":"success", "score": prob_english_preds}

# for testing locally
if __name__ == "__main__":
    classify_accent("../data/", "../binary_accent_classifier.onnx")