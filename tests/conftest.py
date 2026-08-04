"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import Vocabulary, RoCoCaptionDataset
from src.swin_transformer import SwinCaptioner


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_vocabulary():
    """Create a small vocabulary for testing."""
    return Vocabulary(
        token_to_id={
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "the": 4,
            "cat": 5,
            "dog": 6,
            "sits": 7,
            "on": 8,
            "mat": 9,
        },
        id_to_token=["<pad>", "<unk>", "<bos>", "<eos>", "the", "cat", "dog", "sits", "on", "mat"],
        document_frequency={
            "the": 5,
            "cat": 3,
            "dog": 2,
            "sits": 2,
            "on": 2,
            "mat": 1,
        },
    )


@pytest.fixture
def test_dataset_dir(temp_dir, sample_vocabulary):
    """Create a minimal test dataset with images and captions."""
    # Create directory structure
    train_dir = temp_dir / "train_images" / "train"
    valid_dir = temp_dir / "valid_images" / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy images
    image_size = 224
    for split, dir_path in [("train", train_dir), ("valid", valid_dir)]:
        for i in range(3):
            img = Image.fromarray(np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8))
            img.save(dir_path / f"image_{i:05d}.jpg")

    # Create dummy captions CSV
    train_csv = temp_dir / "train_captions.csv"
    train_lines = ["ID,Caption\n"]
    for i in range(3):
        train_lines.append(f"image_{i:05d},the cat sits on mat\n")
    train_csv.write_text("".join(train_lines))

    valid_csv = temp_dir / "valid_captions.csv"
    valid_lines = ["ID,Caption\n"]
    for i in range(3):
        valid_lines.append(f"image_{i:05d},the dog sits on mat\n")
    valid_csv.write_text("".join(valid_lines))

    # Save vocabulary
    sample_vocabulary.save(temp_dir / "vocabulary.json")

    yield temp_dir


@pytest.fixture
def swin_captioner():
    """Create a small SwinCaptioner model for testing."""
    vocab_size = 10
    visual_dim = 64
    embedding_dim = 64
    hidden_dim = 128
    return SwinCaptioner(
        vocab_size=vocab_size,
        visual_dim=visual_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        pad_id=0,
        image_size=224,
    )


@pytest.fixture
def sample_batch():
    """Create a sample batch of images and captions."""
    batch_size = 2
    image_size = 224
    seq_length = 16

    images = torch.randn(batch_size, 3, image_size, image_size)
    captions = torch.randint(0, 10, (batch_size, seq_length))

    return images, captions


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
