# TransenseCSI

## Introduction
This repository contains the implementation code for TransenseCSI.

## Dependencies
- Python 3.8
- PyTorch 2.4.1
- NumPy 1.19.5

## Data Preparation

1. Download the data from [this link](https://cuhko365-my.sharepoint.com/:f:/g/personal/220019036_link_cuhk_edu_cn/EkZPCkqR78tCvAMFt3hhOzMBcysGvRYG_1s1ey13l7AJoA?e=pNWTzp).

2. Place the downloaded data in the root directory as `ROOT_DIRECTORY/data/data_DCSnn/data_DCSnn.mat`

## Training and Evaluation

```bash
python train.py --NN LSTM --T 10 # Train LSTM
python train.py --NN Atten --T 10 # Train TransenseCSI
python quant_train.py --NN Atten --T 10 --src_exp_dir exps --n_bit 4 --quantize uniform # Uniform quantization
python quant_train.py --NN Atten --T 10 --src_exp_dir exps --n_bit 4 --quantize opt # Proposed alternating quantization

