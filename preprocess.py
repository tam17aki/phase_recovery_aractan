# -*- coding: utf-8 -*-
"""Proprocess script: resampling, split and feature extraction.

Copyright (C) 2024 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import glob
import math
import os
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import soundfile as sf
from pydub.audio_segment import AudioSegment
from scipy import signal
from scipy.signal.windows import get_window
from tqdm import tqdm

import config


def resample_wav(sample_rate: int = 44100, is_train: bool = True) -> None:
    """Resample wav file.

    Notice:
        The original audio files must be put in the data_dir/"orig",
        e.g., /work/tamamori/rfa/data/basic5000/orig/

    Args:
        sample_rate (int): sampling rate of the original audio.
        is_train (bool): handling training dataset or test dataset.
    """
    path_cfg = config.PathConfig()
    preproc_cfg = config.PreProcessConfig()
    feat_cfg = config.FeatureConfig()
    dataset_dir = path_cfg.trainset_dir if is_train is True else path_cfg.evalset_dir
    wav_dir = os.path.join(path_cfg.root_dir, path_cfg.data_dir, dataset_dir, "orig")
    resample_dir = os.path.join(
        path_cfg.root_dir, path_cfg.data_dir, dataset_dir, preproc_cfg.resample_dir
    )
    wav_list = os.listdir(wav_dir)  # basename
    wav_list.sort()

    os.makedirs(resample_dir, exist_ok=True)
    for wav_name in tqdm(
        wav_list,
        desc="Resampling wave files",
        bar_format="{desc}: {percentage:3.0f}% ({n_fmt} of {total_fmt}) |{bar}|"
        + " Elapsed Time: {elapsed} ETA: {remaining} ",
        ascii=" #",
    ):
        wav_path = os.path.join(wav_dir, wav_name)
        wav, rate = sf.read(wav_path)
        if rate != sample_rate:
            down_sampled = librosa.resample(
                wav, orig_sr=rate, target_sr=feat_cfg.sample_rate
            )
        else:
            down_sampled = wav
        out_path = os.path.join(resample_dir, wav_name)
        sf.write(out_path, down_sampled, feat_cfg.sample_rate, subtype="PCM_16")


def split_utterance() -> None:
    """Split utterances after resampling into segments."""
    path_cfg = config.PathConfig()
    preproc_cfg = config.PreProcessConfig()
    wav_dir = os.path.join(
        path_cfg.root_dir,
        path_cfg.data_dir,
        path_cfg.trainset_dir,
        preproc_cfg.resample_dir,
    )
    wav_list = glob.glob(wav_dir + "/*.wav")
    wav_list.sort()
    sec_per_split = preproc_cfg.sec_per_split

    out_dir = os.path.join(
        path_cfg.root_dir,
        path_cfg.data_dir,
        path_cfg.trainset_dir,
        preproc_cfg.split_dir,
    )
    os.makedirs(out_dir, exist_ok=True)
    for wav_name in tqdm(
        wav_list,
        desc="Splitting utterances",
        bar_format="{desc}: {percentage:3.0f}% ({n_fmt} of {total_fmt}) |{bar}|"
        + " Elapsed Time: {elapsed} ETA: {remaining} ",
        ascii=" #",
    ):
        audio = AudioSegment.from_wav(wav_name)
        duration = math.floor(audio.duration_seconds)
        for i in range(0, int(duration // sec_per_split)):
            basename, ext = os.path.splitext(wav_name)
            split_fn = basename + "_" + str(i) + ext
            out_file = os.path.join(out_dir, os.path.basename(split_fn))
            split_audio = audio[i * 1000 : (i + sec_per_split) * 1000]
            if split_audio.duration_seconds > (sec_per_split - 0.01):
                split_audio.export(out_file, format="wav")


def _extract_feature(utt_id: str, feat_dir: str, is_train: bool) -> None:
    """Perform feature extraction.

    Args:
        utt_id (str): basename for audio.
        feat_dir (str): directory name for saving features.
        is_train (bool): handling training dataset or test dataset.
    """
    path_cfg = config.PathConfig()
    feat_cfg = config.FeatureConfig()
    preproc_cfg = config.PreProcessConfig()
    if is_train is True:
        wav_dir = os.path.join(
            path_cfg.root_dir,
            path_cfg.data_dir,
            path_cfg.trainset_dir,
            preproc_cfg.split_dir,
        )
    else:
        wav_dir = os.path.join(
            path_cfg.root_dir,
            path_cfg.data_dir,
            path_cfg.evalset_dir,
            preproc_cfg.resample_dir,
        )
    wav_file = os.path.join(wav_dir, utt_id + ".wav")
    audio, rate = sf.read(wav_file)
    if audio.dtype in [np.int16, np.int32]:
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float64)  # type:ignore
    audio = audio.astype(np.float64)

    stfft = signal.ShortTimeFFT(
        win=get_window(feat_cfg.window, feat_cfg.win_length),
        hop=feat_cfg.hop_length,
        fs=rate,
        mfft=feat_cfg.n_fft,
        phase_shift=None,
    )
    stft_data = stfft.stft(audio)
    np.save(
        os.path.join(feat_dir, f"{utt_id}-feats_logamp.npy"),
        np.log(np.abs(stft_data) + 1e-10).astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        os.path.join(feat_dir, f"{utt_id}-feats_phase.npy"),
        np.angle(stft_data).astype(np.float32),
        allow_pickle=False,
    )


def extract_feature(is_train: bool = True) -> None:
    """Extract acoustic features.

    Args:
        is_train (bool): handling training dataset or test dataset.
    """
    path_cfg = config.PathConfig()
    preproc_cfg = config.PreProcessConfig()
    feat_cfg = config.FeatureConfig()
    dataset_dir = path_cfg.trainset_dir if is_train is True else path_cfg.evalset_dir
    wav_dir = preproc_cfg.split_dir if is_train is True else preproc_cfg.resample_dir
    wav_list = os.listdir(
        os.path.join(path_cfg.root_dir, path_cfg.data_dir, dataset_dir, wav_dir)
    )
    utt_list = [
        os.path.splitext(os.path.basename(wav_file))[0] for wav_file in wav_list
    ]
    utt_list.sort()

    feat_dir = os.path.join(
        path_cfg.root_dir, feat_cfg.feat_dir, dataset_dir, feat_cfg.window
    )
    os.makedirs(feat_dir, exist_ok=True)

    with ProcessPoolExecutor(preproc_cfg.n_jobs) as executor:
        futures = [
            executor.submit(_extract_feature, utt, feat_dir, is_train)
            for utt in utt_list
        ]
        for future in tqdm(
            futures,
            desc="Extracting acoustic features",
            bar_format="{desc}: {percentage:3.0f}% ({n_fmt} of {total_fmt}) |{bar}|"
            + " Elapsed Time: {elapsed} ETA: {remaining} ",
            ascii=" #",
        ):
            future.result()  # return None


def main():
    """Perform preprocess."""
    # training data
    resample_wav()
    split_utterance()
    extract_feature()

    # test data
    resample_wav(is_train=False)
    extract_feature(is_train=False)


if __name__ == "__main__":
    main()
