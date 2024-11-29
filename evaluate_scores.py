# -*- coding: utf-8 -*-
"""Evaluation script for sound quality based on PESQ, STOI and LSC.

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

import os
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi
from scipy import signal
from torch.multiprocessing import set_start_method
from tqdm import tqdm

import config
from model import PhaseRecoveryNet


def load_checkpoint() -> PhaseRecoveryNet:
    """Load checkpoint.

    Args:
        None.

    Returns:
        model (PhaseRecoveryNet): DNN to estimate phase.
    """
    cfg = config.PathConfig()
    model_cfg = config.ModelConfig()
    model_dir = os.path.join(cfg.root_dir, "model")
    model = PhaseRecoveryNet().cuda()
    model_file = os.path.join(model_dir, model_cfg.model_file + ".pth")
    checkpoint = torch.load(model_file, weights_only=True)
    model.load_state_dict(checkpoint)
    return model


def get_wavdir() -> str:
    """Return dirname of wavefile to be evaluated.

    Args:
        None.

    Returns:
        wav_dir (str): dirname of wavefile.
    """
    path_cfg = config.PathConfig()
    eval_cfg = config.EvalConfig()
    wav_dir = os.path.join(path_cfg.root_dir, eval_cfg.demo_dir)
    return wav_dir


def get_wavname(basename: str) -> str:
    """Return filename of wavefile to be evaluated.

    Args:
        basename (str): basename of wavefile for evaluation.

    Returns:
        wav_file (str): filename of wavefile.
    """
    wav_name, _ = os.path.splitext(basename)
    wav_name = wav_name.split("-")[0]
    wav_dir = get_wavdir()
    wav_file = os.path.join(wav_dir, wav_name + ".wav")
    return wav_file


def compute_pesq(basename: str) -> float:
    """Compute PESQ and wideband PESQ.

    Args:
        basename (str): basename of wavefile for evaluation.

    Returns:
        float: PESQ (or wideband PESQ).
    """
    cfg = config.PathConfig()
    eval_cfg = config.EvalConfig()
    preproc_cfg = config.PreProcessConfig()
    eval_wav, rate = sf.read(get_wavname(basename))
    eval_wav = librosa.resample(eval_wav, orig_sr=rate, target_sr=16000)
    ref_wavname, _ = os.path.splitext(basename)
    ref_wavname = ref_wavname.split("-")[0]  # remove '_logamp'
    wav_dir = os.path.join(
        cfg.root_dir, cfg.data_dir, cfg.evalset_dir, preproc_cfg.resample_dir
    )
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    if rate != 16000:
        reference = librosa.resample(y=reference, orig_sr=rate, target_sr=16000)
    if eval_wav.size > reference.size:
        eval_wav = eval_wav[: reference.size]
    else:
        reference = reference[: eval_wav.size]
    return float(pesq(16000, reference, eval_wav, eval_cfg.pesq_mode))


def compute_stoi(basename: str) -> float:
    """Compute STOI or extended STOI (ESTOI).

    Args:
        basename (str): basename of wavefile for evaluation.

    Returns:
        float: STOI (or ESTOI).
    """
    cfg = config.PathConfig()
    eval_cfg = config.EvalConfig()
    preproc_cfg = config.PreProcessConfig()
    eval_wav, _ = sf.read(get_wavname(basename))
    ref_wavname, _ = os.path.splitext(basename)
    ref_wavname = ref_wavname.split("-")[0]  # remove '_logamp'
    wav_dir = os.path.join(
        cfg.root_dir, cfg.data_dir, cfg.evalset_dir, preproc_cfg.resample_dir
    )
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    if eval_wav.size > reference.size:
        eval_wav = eval_wav[: reference.size]
    else:
        reference = reference[: eval_wav.size]
    return float(stoi(reference, eval_wav, rate, extended=eval_cfg.stoi_extended))


def compute_lsc(basename: str) -> np.float64:
    """Compute log-spectral convergence (LSC).

    Args:
        basename (str): basename of wavefile for evaluation.

    Returns:
        lsc (float64): log-spectral convergence.
    """
    cfg = config.PathConfig()
    feat_cfg = config.FeatureConfig()
    preproc_cfg = config.PreProcessConfig()
    eval_wav, _ = sf.read(get_wavname(basename))
    ref_wavname, _ = os.path.splitext(basename)
    ref_wavname = ref_wavname.split("-")[0]  # remove '_logamp'
    wav_dir = os.path.join(
        cfg.root_dir, cfg.data_dir, cfg.evalset_dir, preproc_cfg.resample_dir
    )
    reference, rate = sf.read(os.path.join(wav_dir, ref_wavname + ".wav"))
    if eval_wav.size > reference.size:
        eval_wav = eval_wav[: reference.size]
    else:
        reference = reference[: eval_wav.size]
    stfft = signal.ShortTimeFFT(
        win=signal.get_window(feat_cfg.window, feat_cfg.win_length),
        hop=feat_cfg.hop_length,
        fs=rate,
        mfft=feat_cfg.n_fft,
        phase_shift=None,
    )
    ref_abs = np.abs(stfft.stft(reference))
    eval_abs = np.abs(stfft.stft(eval_wav))
    lsc = np.linalg.norm(ref_abs - eval_abs)
    lsc = lsc / np.linalg.norm(ref_abs)
    lsc = 20 * np.log10(lsc)
    return lsc


@torch.no_grad()
def recover_phase(
    model: PhaseRecoveryNet, logamp: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Recover phase spectrum.

    Args:
        model (PhaseRecoveryNet): DNN (nn.Module).
        logamp (ndarray): log-amplitude spectrum. [F, T]

    Returns:
        phase (ndarray): reconstruced phase. [F, T]
    """
    logamp_tensor = torch.tensor(logamp).float().unsqueeze(0).cuda()  # [1, T+L+1, K]
    phase = model(logamp_tensor)
    phase = phase.to("cpu").detach().numpy().copy()
    phase = np.squeeze(phase)
    return phase


def _reconst_waveform(model: PhaseRecoveryNet, logamp_path: str) -> None:
    """Reconstruct audio waveform only from the amplitude spectrum.

    Args:
        model (PhaseRecoveryNet): DNN params (nn.Module).
        logamp_path (str): path to the log-amplitude spectrum.

    Returns:
        None.
    """
    cfg = config.FeatureConfig()
    logamp = np.load(logamp_path)  # [F, T]
    phase = recover_phase(model, logamp)
    spec = np.exp(logamp + 1j * phase)  # [F, T]
    stfft = signal.ShortTimeFFT(
        win=signal.get_window(cfg.window, cfg.win_length),
        hop=cfg.hop_length,
        fs=cfg.sample_rate,
        mfft=cfg.n_fft,
        phase_shift=None,
    )
    audio = stfft.istft(spec)
    wav_file = get_wavname(os.path.basename(logamp_path))
    sf.write(wav_file, audio, cfg.sample_rate)


def reconst_waveform(model: PhaseRecoveryNet, logamp_list: list[str]) -> None:
    """Reconstruct audio waveforms in parallel.

    Args:
        model (PhaseRecoveryNet): DNN params (nn.Module).
        logamp_list (list): list of path to the log-amplitude spectrum.

    Returns:
        None.
    """
    cfg = config.PreProcessConfig()
    set_start_method("spawn")
    with ProcessPoolExecutor(cfg.n_jobs) as executor:
        futures = [
            executor.submit(_reconst_waveform, model, logamp_path)
            for logamp_path in logamp_list
        ]
        for future in tqdm(
            futures,
            desc="Reconstruct waveform",
            bar_format="{desc}: {percentage:3.0f}% ({n_fmt} of {total_fmt}) |{bar}|"
            " Elapsed Time: {elapsed} ETA: {remaining} ",
            ascii=" #",
        ):
            future.result()  # return None


def _compuate_accuracy(
    model: PhaseRecoveryNet, logamp_path: str, phase_path: str
) -> float:
    """Compute accuracy of phase estimation.

    Args:
        model (PhaseRecoveryNet): DNN (nn.Module).
        logamp_path (str): path to the log-amplitude spectrum.
        phase_path (str): path to the phase spectrum.

    Returns:
        acc (ndarray): accuracy of phase estimation (= scaler value).
    """
    logamp = np.load(logamp_path)  # [K, T]
    true_phase = np.load(phase_path)  # [K, T]
    pred_phase = recover_phase(model, logamp)
    pred_phase = np.squeeze(pred_phase)
    acc = np.mean(np.cos(pred_phase - true_phase))
    return acc


def compute_accuracy(
    model: PhaseRecoveryNet, logamp_list: list[str], phase_list: list[str]
) -> None:
    """Compute estimation accuracy in parallel.

    Args:
        model (PhaseRecoveryNet): DNN params (nn.Module).
        logamp_list (list): list of path to the log-amplitude spectrum.
        phase_list (list): list of path to the phase spectrum.

    Returns:
        None.
    """
    score = []
    for logamp_path, phase_path in zip(logamp_list, phase_list):
        score.append(_compuate_accuracy(model, logamp_path, phase_path))
    acc = np.array(score).mean()
    print(f"Accuracy of phase estimation = {acc:.6f}")


def compute_obj_scores(logamp_list: list[str]) -> dict[str, list[np.float64 | float]]:
    """Compute objective scores; PESQ, STOI and LSC.

    Args:
        logamp_list (list): list of path to the log-amplitude spectrum.

    Returns:
        score_dict (dict): dictionary of objective score lists.
    """
    score_dict: dict[str, list[np.float64 | float]] = {
        "pesq": [],
        "stoi": [],
        "lsc": [],
    }
    for logamp_path in tqdm(
        logamp_list,
        desc="Compute objective scores",
        bar_format="{desc}: {percentage:3.0f}% ({n_fmt} of {total_fmt}) |{bar}|"
        " Elapsed Time: {elapsed} ETA: {remaining} ",
        ascii=" #",
    ):
        score_dict["pesq"].append(compute_pesq(os.path.basename(logamp_path)))
        score_dict["stoi"].append(compute_stoi(os.path.basename(logamp_path)))
        score_dict["lsc"].append(compute_lsc(os.path.basename(logamp_path)))
    return score_dict


def aggregate_scores(
    score_dict: dict[str, list[np.float64 | float]], score_dir: str
) -> None:
    """Aggregate objective evaluation scores.

    Args:
        score_dict (dict): dictionary of objective score lists.
        score_dir (str): dictionary name of objective score files.

    Returns:
        None.
    """
    for score_type, score_list in score_dict.items():
        out_filename = f"{score_type}_score.txt"
        out_filename = os.path.join(score_dir, out_filename)
        with open(out_filename, mode="w", encoding="utf-8") as file_handler:
            for score in score_list:
                file_handler.write(f"{score}\n")
        score_array = np.array(score_list)
        print(
            f"{score_type}: "
            f"mean={np.mean(score_array):.6f}, "
            f"median={np.median(score_array):.6f}, "
            f"std={np.std(score_array):.6f}, "
            f"max={np.max(score_array):.6f}, "
            f"min={np.min(score_array):.6f}"
        )


def load_logamp(is_train=False) -> list[str]:
    """Load file paths for log-amplitude spectrogram.

    Args:
        is_train (bool) : boolean

    Returns:
        logamp_list (list): list of file path for log-amplitude spectrogram.
    """
    path_cfg = config.PathConfig()
    feat_cfg = config.FeatureConfig()
    preproc_cfg = config.PreProcessConfig()
    if is_train:
        feat_dir = os.path.join(
            path_cfg.root_dir, feat_cfg.feat_dir, path_cfg.trainset_dir, feat_cfg.window
        )
        wav_list = os.listdir(
            os.path.join(
                path_cfg.root_dir,
                path_cfg.data_dir,
                path_cfg.trainset_dir,
                preproc_cfg.split_dir,
            )
        )
    else:
        feat_dir = os.path.join(
            path_cfg.root_dir, feat_cfg.feat_dir, path_cfg.evalset_dir, feat_cfg.window
        )
        wav_list = os.listdir(
            os.path.join(
                path_cfg.root_dir,
                path_cfg.data_dir,
                path_cfg.evalset_dir,
                preproc_cfg.resample_dir,
            )
        )
    utt_list = [
        os.path.splitext(os.path.basename(wav_file))[0] for wav_file in wav_list
    ]
    utt_list.sort()
    logamp_list = []
    for utt_id in utt_list:
        logamp_list.append(os.path.join(feat_dir, f"{utt_id}-feats_logamp.npy"))
    logamp_list.sort()
    return logamp_list


def load_phase(is_train=False) -> list[str]:
    """Load file paths for phase spectrogram.

    Args:
        is_train (bool): boolean

    Returns:
        phase_list (list): list of file path for phase spectrogram.
    """
    path_cfg = config.PathConfig()
    feat_cfg = config.FeatureConfig()
    preproc_cfg = config.PreProcessConfig()
    if is_train:
        feat_dir = os.path.join(
            path_cfg.root_dir, feat_cfg.feat_dir, path_cfg.trainset_dir, feat_cfg.window
        )
        wav_list = os.listdir(
            os.path.join(
                path_cfg.root_dir,
                path_cfg.data_dir,
                path_cfg.trainset_dir,
                preproc_cfg.split_dir,
            )
        )
    else:
        feat_dir = os.path.join(
            path_cfg.root_dir, feat_cfg.feat_dir, path_cfg.evalset_dir, feat_cfg.window
        )
        wav_list = os.listdir(
            os.path.join(
                path_cfg.root_dir,
                path_cfg.data_dir,
                path_cfg.evalset_dir,
                preproc_cfg.resample_dir,
            )
        )
    utt_list = [
        os.path.splitext(os.path.basename(wav_file))[0] for wav_file in wav_list
    ]
    utt_list.sort()
    phase_list = []
    for utt_id in utt_list:
        phase_list.append(os.path.join(feat_dir, f"{utt_id}-feats_phase.npy"))
    phase_list.sort()
    return phase_list


def main() -> None:
    """Perform evaluation."""
    # setup directory
    cfg = config.PathConfig()
    eval_cfg = config.EvalConfig()
    wav_dir = get_wavdir()
    os.makedirs(wav_dir, exist_ok=True)
    score_dir = os.path.join(cfg.root_dir, eval_cfg.score_dir)
    os.makedirs(score_dir, exist_ok=True)

    # load DNN parameters
    model = load_checkpoint()
    model.eval()

    # load list of file paths for log-amplitude spectrogram
    logamp_list = load_logamp()

    # reconstruct phase and waveform in parallel
    reconst_waveform(model, logamp_list)

    # compute objective scores
    score_dict = compute_obj_scores(logamp_list)

    # aggregate objective scores
    aggregate_scores(score_dict, score_dir)

    # load list of file paths for phase spectrogram
    phase_list = load_phase()

    # compute estimation accuracy
    compute_accuracy(model, logamp_list, phase_list)


if __name__ == "__main__":
    main()
