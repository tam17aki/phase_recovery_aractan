# -*- coding: utf-8 -*-
"""Training script for Phase Recovery.

Copyright (C) 2024 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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

import torch
from torch import nn
from torchinfo import summary
from tqdm import tqdm

import config
from dataset import get_dataloader
from factory import get_loss, get_optimizer
from model import get_model


def main() -> None:
    """Perform model training."""
    # dump configs
    for cfg in (
        config.PathConfig(),
        config.PreProcessConfig(),
        config.FeatureConfig(),
        config.ModelConfig(),
        config.OptimizerConfig(),
        config.TrainingConfig(),
        config.EvalConfig(),
    ):
        print(cfg)
    cfg = config.TrainingConfig()
    path_cfg = config.PathConfig()
    model_cfg = config.ModelConfig()

    # instantiate modules
    dataloader = get_dataloader()
    model = get_model().cuda()
    loss_func = get_loss(model)
    optimizer = get_optimizer(model)

    # print summary of the network architecture and the number of parameters.
    _ = summary(model, depth=4)

    # perform training loop
    _ = model.train()
    optimizer.train()
    for epoch in tqdm(
        range(1, cfg.n_epoch + 1),
        desc="Model training",
        bar_format="{desc}: {percentage:3.0f}% ({n_fmt} of {total_fmt}) |{bar}|"
        + " Elapsed Time: {elapsed} ETA: {remaining} ",
        ascii=" #",
    ):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_func(batch)
            epoch_loss += loss.item()
            loss.backward()
            if cfg.use_grad_clip:
                _ = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_max_norm)
            _ = optimizer.step()
        epoch_loss = epoch_loss / len(dataloader)
        if epoch == 1 or epoch % cfg.report_interval == 0:
            print(f"\nEpoch {epoch}: loss={epoch_loss:.12f}")

    # save model
    model_dir = os.path.join(path_cfg.root_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    optimizer.eval()
    torch.save(
        model.state_dict(), f=os.path.join(model_dir, model_cfg.model_file + ".pth")
    )


if __name__ == "__main__":
    main()
