# SAGE: Semantic-Aware Tokenization with Ranking Guidance for Generative Recommendation

This repository contains reproduction code for the paper **“SAGE: Semantic-Aware Tokenization with Ranking Guidance for Generative Recommendation.”**

## Overview

This project runs sequence recommendation experiments on the **Amazon Reviews 2014** dataset with the **genrec** framework and the **SAGE** model.

## Main Components

- `main.py`: entry point for experiments.
- `genrec/`: general recommendation framework (pipeline, training, evaluation, tokenization, etc.).
- `genrec/datasets/AmazonReviews2014/`: dataset loading and preprocessing logic for Amazon Reviews 2014.
- `genrec/model.py` and `genrec/models/SAGE/model.py`: sequence recommendation model interfaces and SAGE model implementation.
- `cache/`: cached and preprocessed data; if you regenerate data locally, old cache files do not need to be kept.

## Environment

Use a Python virtual environment:

```bash
python -m venv .venv
```

On Windows, activate it with:

```powershell
.venv\Scripts\activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## Data

By default, the repository expects processed data under:

- `cache/AmazonReviews2014/`

If data is missing, download and preprocess Amazon Reviews 2014 following the original paper setup or your local data pipeline.

## Basic Run Hint

`main.py` is the experiment entry. A basic run example is:

## Quick start
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Sports_and_Outdoors
```
## Reproduction
### Sports and Outdoors
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --category=Sports_and_Outdoors \
    --rand_seed=2026 \
    --weight_decay=0.15 \
    --lr=0.005 \
    --item_weight_gamma=2.0 \
    --item_weight_clip_range="[0.2,5.0]" \
    --ranking_temperature=0.7 \
    --n_hash_buckets=128
```
### Beauty
```bash
for seed in {2026..2028}; do
  CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --category=Beauty \
  --rand_seed=${seed} \
  --weight_decay=0.15 \
  --lr=0.001 \
  --n_hash_buckets=64 \
  --item_weight_gamma=3.0 \
  --item_weight_clip_range="[0.2,4.0]" \
  --ranking_temperature=1.0done
done
```
### CDs and Vinyl
```bash
for seed in {2024..2028}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --category=CDs_and_Vinyl \
        --rand_seed=${seed} \
        --weight_decay=0.07 \
        --lr=0.001 \
        --d_model=256 \
        --d_ff=2048 \
        --n_hash_buckets=256 \
        --item_weight_gamma=2.5 \
        --item_weight_clip_range="[0.3,3.5]" \
        --ranking_temperature=0.7
done
```
