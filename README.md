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

## Data

The repository assumes Amazon Reviews 2014 data is available under `cache/AmazonReviews2014/`. If these files are not present, please download and preprocess the dataset following the instructions in the original paper or your own data pipeline.

