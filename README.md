# Swin Transformer starter project

This folder contains a lightweight Swin-style transformer starter codebase for the biomedical image-captioning project.

## Structure
- `src/swin_transformer.py`: a compact Swin-inspired transformer encoder and classifier
- `src/data.py`: a simple dataset wrapper for image folders
- `train.py`: a training entry point

## Quick start
```bash
cd "swin transformer"
python3 -m pip install -r requirements.txt
python3 train.py --data-root /path/to/images --epochs 1 --batch-size 4
```

> This is a starter implementation intended for experimentation and extension.
