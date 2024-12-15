# -*- coding: utf-8 -*-
"""A Python module providing dataset and dataloader.

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
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from torch.utils.data import DataLoader, Dataset

import config


@dataclass
class FeatPath:
    """Paths for features."""

    logamp: list[str]
    phase: list[str]


class PhaseRecoveryDataset(
    Dataset[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]
):
    """Dataset for training neural net."""

    def __init__(self, feat_paths: FeatPath) -> None:
        """Initialize class."""
        self.logamp_paths: list[str] = feat_paths.logamp
        self.phase_paths: list[str] = feat_paths.phase

    def __len__(self) -> int:
        """Return the size of the dataset.

        Returns:
            int: size of the dataset
        """
        return len(self.logamp_paths)

    def __getitem__(
        self, idx: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Get a pair of input and target.

        Args:
            idx (int): index of the pair

        Returns:
            Tuple: input and target in numpy format
        """
        return (np.load(self.logamp_paths[idx]), np.load(self.phase_paths[idx]))


def get_dataloader() -> (
    DataLoader[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]]
):
    """Get a dataloader for training.

    Returns:
        dataloader (Dataloader): a dataloader for training.
    """
    train_cfg = config.TrainingConfig()
    path_cfg = config.PathConfig()
    feat_cfg = config.FeatureConfig()
    preproc_cfg = config.PreProcessConfig()
    wav_list = os.listdir(
        os.path.join(
            path_cfg.root_dir,
            path_cfg.data_dir,
            path_cfg.trainset_dir,
            preproc_cfg.split_dir,
        )
    )
    utt_list = [
        os.path.splitext(os.path.basename(wav_file))[0] for wav_file in wav_list
    ]
    utt_list.sort()

    feat_dir = os.path.join(
        path_cfg.root_dir, feat_cfg.feat_dir, path_cfg.trainset_dir, feat_cfg.window
    )
    feat_paths = FeatPath(
        logamp=[
            os.path.join(feat_dir, f"{utt_id}-feats_logamp.npy") for utt_id in utt_list
        ],
        phase=[
            os.path.join(feat_dir, f"{utt_id}-feats_phase.npy") for utt_id in utt_list
        ],
    )
    dataloader = DataLoader(
        PhaseRecoveryDataset(feat_paths),
        batch_size=train_cfg.n_batch,
        pin_memory=True,
        num_workers=train_cfg.num_workers,
        shuffle=True,
        drop_last=True,
    )
    return dataloader
