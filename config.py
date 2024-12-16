# -*- coding: utf-8 -*-
"""Config script.

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

from dataclasses import dataclass


@dataclass(frozen=True)
class PathConfig:
    """Dataclass for path config."""

    root_dir: str = "/work/tamamori/phase_recovery_arctan/"
    data_dir: str = "data/"
    trainset_dir: str = "basic5000/"
    evalset_dir: str = "onoma300/"


@dataclass(frozen=True)
class PreProcessConfig:
    """Dataclass for preprocess."""

    resample_dir: str = "resample_16k/"
    split_dir: str = "split/"
    n_jobs: int = 6
    sec_per_split: float = 1.0  # seconds per split of a audio clip


@dataclass(frozen=True)
class FeatureConfig:
    """Dataclass for feature extraction."""

    feat_dir: str = "feat/"
    sample_rate: int = 16000
    win_length: int = 512
    hop_length: int = 128
    window: str = "hann"
    n_fft: int = 512


@dataclass(frozen=True)
class ModelConfig:
    """Dataclass for model definition."""

    input_channels: int = 257
    hidden_channels: int = 128
    hidden_channels_post: int = 128
    kernel_size: int = 5
    n_resblock: int = 10
    n_postblock: int = 10
    model_file: str = "model"


@dataclass(frozen=True)
class OptimizerConfig:
    """Dataclass for optimizer."""

    name: str = "RAdam"
    lr: float = 0.001
    weight_decay: float = 0.1
    decoupled_weight_decay: bool = True


@dataclass(frozen=True)
class SchedulerConfig:
    """Dataclass for learning rate scheduler."""

    name: str = "CosineLRScheduler"
    warmup_t: int = 5
    t_initial: int = 1000  # total number of epochs
    warmup_lr_init: float = 0.000001
    lr_min: float = 0.00001  # final learning rate


@dataclass(frozen=True)
class TrainingConfig:
    """Dataclass for training."""

    n_epoch: int = 1000
    n_batch: int = 64
    num_workers: int = 1
    report_interval: int = 10
    use_scheduler: bool = True
    use_grad_clip: bool = True
    grad_max_norm: float = 10.0


@dataclass(frozen=True)
class EvalConfig:
    """Dataclass for evaluation."""

    pesq_mode: str = "wb"  # "wb": wideband or "nb": narrowband
    stoi_extended: bool = True  # True: extended STOI
    demo_dir: str = "demo/"
    score_dir: str = "score/"
