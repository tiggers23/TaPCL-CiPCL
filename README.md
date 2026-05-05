# Efficiently Maintaining the Multilingual Capacity of MCLIP in Downstream Cross-Modal Retrieval Tasks

This repository provides the code for **Efficiently Maintaining the Multilingual Capacity of MCLIP in Downstream Cross-Modal Retrieval Tasks**. The project studies how to fine-tune Multilingual CLIP (MCLIP) for downstream image-text retrieval while reducing the training cost caused by full parallel corpora.

The repository mainly supports two methods:

- **TaPCL**: Transferability-aware Parallel Corpora Learning. It uses token-overlap based sampling to assign higher sampling probability to target languages that are less similar to the source language.
- **CiPCL**: Critical-information Parallel Corpora Learning. It uses translated key terms instead of full target-language parallel sentences to reduce the use of complete parallel corpora.

## Requirements

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Data

The dataset can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/18oWhvtJH5u0X0X1P3wZgoA?pwd=bxcj).

Extraction code: `bxcj`

After downloading, please place the dataset under:

```bash
DATA/
```

## Pretrained Models

The code loads the OpenCLIP ViT-B/32 XLM-Roberta model by default from:

```bash
Pretrained_models/OpenCLIP-ViT-B-32-xlm-roberta-base/pytorch_model.bin
```

For the translation script, the code loads the M2M100 model by default from:

```bash
Pretrained_models/facebookm2m100_418M
```
Users can download these pretrained models from Hugging Face and place them in the specified folders.

## Running

First, generate the translated key-term prompt data for CiPCL if needed:

```bash
bash run_translate.sh
```

Run TaPCL with:

```bash
bash run_TaPCL.sh
```

Run CiPCL with:

```bash
bash run_CiPCL.sh
```

## Note

Some paths in the scripts are dataset-specific. If you use another dataset or another machine, please modify `dataset_root`, `class_order`, `pretrained model paths`, and the output path in `models_translate.py` accordingly.
