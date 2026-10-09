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

```bash
python main.py --model SAGE --dataset AmazonReviews2014
```
