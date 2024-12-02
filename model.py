# -*- coding: utf-8 -*-
"""A Python module providing a neural network for phase recovery.

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

import torch
from torch import nn

import config


class PreNet(nn.Module):
    """PreNet module."""

    def __init__(self):
        """Initialize class."""
        super().__init__()
        cfg = config.ModelConfig()
        in_channels = cfg.input_channels
        hid_channels = cfg.hidden_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hid_channels, 1),
            nn.GELU(),
            nn.Conv1d(hid_channels, hid_channels, 1),
        )

    def forward(self, inputs) -> torch.Tensor:
        """Forward propagation."""
        inputs = inputs - torch.mean(inputs, dim=-1, keepdim=True)
        return self.net(inputs)


class ResidualBlock(nn.Module):
    """Residual Block module for MiddleNet."""

    def __init__(self):
        """Initialize class."""
        super().__init__()
        cfg = config.ModelConfig()
        hid_channels = cfg.hidden_channels
        kernel_size = cfg.kernel_size
        self.convs = nn.Sequential(  # full pre-activation
            nn.BatchNorm1d(hid_channels),
            nn.GELU(),
            nn.Conv1d(
                hid_channels,
                hid_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm1d(hid_channels),
            nn.GELU(),
            nn.Conv1d(
                hid_channels,
                hid_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
        )

    def forward(self, inputs) -> torch.Tensor:
        """Forward propagation."""
        hidden = self.convs(inputs)
        hidden = hidden + inputs
        return hidden


class MiddleNet(nn.Module):
    """MiddleNet module."""

    def __init__(self):
        """Initialize class."""
        super().__init__()
        cfg = config.ModelConfig()
        resnet = nn.ModuleList([ResidualBlock() for _ in range(cfg.n_resblock)])
        self.resnet = nn.Sequential(*resnet)

    def forward(self, inputs) -> torch.Tensor:
        """Forward propagation."""
        hidden = self.resnet(inputs)
        return hidden


class PostNetBlock(nn.Module):
    """Residual Block module for PostNet."""

    def __init__(self):
        """Initialize class."""
        super().__init__()
        cfg = config.ModelConfig()
        hid_channels = cfg.hidden_channels_post
        kernel_size = cfg.kernel_size
        self.convs = nn.Sequential(  # full pre-activation
            nn.BatchNorm1d(cfg.hidden_channels),
            nn.GELU(),
            nn.Conv1d(
                cfg.hidden_channels,
                hid_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm1d(hid_channels),
            nn.GELU(),
            nn.Conv1d(
                hid_channels,
                hid_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
        )
        self.rnn = nn.GRU(
            hid_channels,
            hid_channels,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Conv1d(2 * hid_channels, cfg.hidden_channels, 1)

    def forward(self, inputs) -> torch.Tensor:
        """Forward propagation."""
        hidden = self.convs(inputs)
        hidden, _ = self.rnn(hidden.transpose(1, 2))
        hidden = self.proj(hidden.transpose(1, 2))
        hidden = hidden + inputs
        return hidden


class PostNet(nn.Module):
    """PostNet module."""

    def __init__(self):
        """Initialize class."""
        super().__init__()
        cfg = config.ModelConfig()
        postnet = nn.ModuleList([PostNetBlock() for _ in range(cfg.n_postblock)])
        self.postnet = nn.Sequential(*postnet)

    def forward(self, inputs) -> torch.Tensor:
        """Forward propagation."""
        outputs = self.postnet(inputs)
        return outputs


class PhaseRecoveryNet(nn.Module):
    """Phase recovery via arctangent with neural network."""

    def __init__(self):
        """Initialize class."""
        super().__init__()
        model_cfg = config.ModelConfig()
        feat_cfg = config.FeatureConfig()
        assert feat_cfg.n_fft // 2 + 1 == model_cfg.input_channels
        self.prenet = PreNet()
        self.midnet = MiddleNet()
        self.postnet = PostNet()
        self.fc_conv = nn.Conv1d(
            model_cfg.hidden_channels, 2 * (feat_cfg.n_fft // 2 + 1), 1
        )

    def forward(self, inputs) -> torch.Tensor:
        """Recover phase from log-amplitude spectrum.

        Args:
            inputs (torch.Tensor): log-amplitude spectrum [B, F, T]

        Returns:
            phase (torch.Tensor): phase spectrum [B, F, T]
        """
        cfg = config.FeatureConfig()
        hidden = self.postnet(self.midnet(self.prenet(inputs)))
        hidden = self.fc_conv(hidden)  # [B, 2F, T]
        n_spec = (cfg.n_fft // 2) + 1
        imag_part = hidden[:, n_spec:, :]  # [B, F, T]
        real_part = hidden[:, :n_spec, :]  # [B, F, T]
        phase = torch.atan2(imag_part, real_part)  # [B, F, T]
        return phase
