# -*- coding: utf-8 -*-
"""A Python module which provides optimizer, scheduler, and customized loss.

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

from typing import override

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn, optim
from torch.optim.optimizer import Optimizer

import config
from model import PhaseRecoveryNet


def get_optimizer(model: PhaseRecoveryNet) -> optim.RAdam | optim.AdamW:
    """Instantiate optimizer.

    Args:
        model (nn.Module): network parameters.

    Returns:
        optimizer (Optimizer): RAdam or AdamW.
    """
    cfg = config.OptimizerConfig()
    if cfg.name == "RAdam":
        return optim.RAdam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            decoupled_weight_decay=cfg.decoupled_weight_decay,
        )
    return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def get_lr_scheduler(optimizer: Optimizer) -> CosineLRScheduler:
    """Instantiate scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.

    Returns:
        lr_scheduler (CosineLRScheduler): cosine scheduler w/ warmup.
    """
    cfg = config.SchedulerConfig()
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=cfg.t_initial,  # total number of epochs
        lr_min=cfg.lr_min,  # final learning rate
        warmup_t=cfg.warmup_t,
        warmup_lr_init=cfg.warmup_lr_init,
        warmup_prefix=True,
    )
    return lr_scheduler


class CustomLoss(nn.Module):
    """Custom loss."""

    def __init__(self, model: PhaseRecoveryNet) -> None:
        """Initialize class.

        Args:
            model (PhaseRecoveryNet): neural network to estimate phase spectrum.
        """
        super(CustomLoss, self).__init__()
        self.model: PhaseRecoveryNet = model

    @override
    def forward(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute loss function.

        Args:
            batch (Tuple): tuple of minibatch.

        Returns:
            loss (Tensor): cosine loss.
        """
        logamp, target_phase = batch
        logamp = logamp.cuda().float()
        target_phase = target_phase.cuda().float()
        pred_phase: torch.Tensor = self.model(logamp)

        target_ifreq = torch.diff(target_phase, dim=2)
        target_grd = -torch.diff(target_phase, dim=1)
        pred_ifreq = torch.diff(pred_phase, dim=2)
        pred_grd = -torch.diff(pred_phase, dim=1)

        loss_cos = -torch.cos(pred_phase - target_phase)
        loss_cos = torch.sum(loss_cos, dim=(1, 2))
        loss_cos_ifreq = -torch.cos(pred_ifreq - target_ifreq)
        loss_cos_ifreq = torch.sum(loss_cos_ifreq, dim=(1, 2))
        loss_cos_grd = -torch.cos(pred_grd - target_grd)
        loss_cos_grd = torch.sum(loss_cos_grd, dim=(1, 2))

        loss = (loss_cos + loss_cos_ifreq + loss_cos_grd).mean()
        return loss


def get_loss(model: PhaseRecoveryNet) -> CustomLoss:
    """Instantiate customized loss.

    Args:
        model (PhaseRecoveryNet): neural network to estimate phase spectrum.

    Returns:
        custom_loss (CustomLoss): customized loss function.
    """
    custom_loss = CustomLoss(model)
    return custom_loss
