# phase_recovery_arctan
This repository provides an implementation of neural network for phase recovery via arctangent.

## Licence
MIT licence.

Copyright (C) 2024 Akira Tamamori

## Dependencies
We tested the implemention on Ubuntu 24.04. The verion of Python was `3.12.3`. The following modules are required:

- joblib
- librosa
- numpy
- pydub
- pypesq
- pystoi
- scikit-learn
- soundfile
- timm
- torch
- tqdm

## Scripts

Mainly, we will use the following scripts:

| Name               | Functionality                                        |
|--------------------|------------------------------------------------------|
| config.py          | Configuration                                        |
| preprocess.py      | Performs pre-processing including feature extraction |
| dataset.py         | Builds the dataset and dataloader                    |
| model.py           | Defines the network architecture                     |
| factory.py         | Defines the optimizer, scheduler, and loss function  |
| training.py        | Trains the model                                     |
| evaluate_scores.py | Calculates objective sound quality metrics           |

## Datasets
You need to prepare the following two datasets from [JSUT corpus](https://sites.google.com/site/shinnosuketakamichi/publication/jsut).

   - basic5000: for training

   - onomatopee300: for evaluation


## Recipes

1. Prepare the two datasets form the JSUT corpus. Put wave files in /root_dir/data_dir/trainset_dir/orig and /root_dir/data_dir/evalset_dir/orig, respectively.

2. Modify `config.py` according to your environment. It contains settings for experimental conditions. For immediate use, you can edit mainly the directory paths according to your environment.

3. Run `preprocess.py`. It performs preprocessing steps.

4. Run `training.py`. It performs model training.

5. Run `evaluate_scores.py`. It generates reconstructed audio data and computes objective scores (PESQ, ESTOI, LSC).

## Notice
The folder `schedulefree` contains a collection of scripts for using RAdamScheduleFree.

