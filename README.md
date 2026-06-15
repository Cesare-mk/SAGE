# SAGE: Semantic-Aware Tokenization with Ranking Guidance for Generative Recommendation

This repository contains the reproduction code for the paper corresponding to this project. 

## Overview

The project implements sequence recommendation experiments on the Amazon Reviews 2014 dataset, using the `genrec` framework and the `SAGE` model.

Main components:
- `main.py`: Entry point for running experiments.
- `genrec/`: General recommendation framework, including dataset and model definitions.
- `genrec/datasets/AmazonReviews2014/`: Dataset loading and preprocessing logic for Amazon Reviews 2014.
- `genrec/models/model.py/`: Implementation of the sequence recommendation model.
- `cache/`: Cached and preprocessed data (not needed to re-run if you regenerate data locally).

## Environment

The code is written in Python. To create a suitable environment, you can use:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

(Adjust the commands if you are on a different platform or using another environment manager.)

## Quick Start

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Sports_and_Outdoors
```

Note that:
1. The datasets will be automatically downloaded once the `category` argument is specified.
2. All hyperparameters can be specified via command line arguments. Please refer to:
    * `genrec/default.yaml`
    * `genrec/datasets/AmazonReviews2014/config.yaml`
    * `genrec/models/SGAE/config.yaml`

## Reproduction

### Sports and Outdoors

Below are the scripts to reproduce the results reported in our paper.

```bash
for seed in {2024..2028}; do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --category=Sports_and_Outdoors \
        --rand_seed=${seed} \
        --weight_decay=0.15 \
        --lr=0.005 \
        --n_hash_buckets=128
done
```

### Beauty

```bash
for seed in {2024..2028}; do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --category=Beauty \
        --rand_seed=${seed} \
        --weight_decay=0.15 \
        --lr=0.001 \
        --n_hash_buckets=64
done
```

### CDs and Vinyl

```bash
for seed in {2024..2028}; do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --category=CDs_and_Vinyl \
        --rand_seed=${seed} \
        --weight_decay=0.07 \
        --lr=0.001 \
        --d_model=256 \
        --d_ff=2048 \
        --n_hash_buckets=256
done
```

