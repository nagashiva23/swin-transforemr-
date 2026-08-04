"""Tests for data module."""

from __future__ import annotations

import csv

import pytest
import torch

from src.data import Vocabulary, RoCoCaptionDataset, tokenize, read_captions


class TestTokenize:
    """Test the tokenize function."""

    def test_simple_tokenization(self):
        """Test basic tokenization."""
        text = "The cat sits on the mat"
        tokens = tokenize(text)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_tokenize_lowercase(self):
        """Test that tokenization produces lowercase tokens."""
        text = "The CAT Sits"
        tokens = tokenize(text)
        assert all(t.islower() or t.isdigit() or not t.isalpha() for t in tokens)

    def test_tokenize_numbers(self):
        """Test tokenization preserves numbers."""
        text = "There are 42 cats"
        tokens = tokenize(text)
        assert "42" in tokens

    def test_tokenize_empty_string(self):
        """Test tokenization of empty string."""
        tokens = tokenize("")
        assert tokens == []


class TestVocabulary:
    """Test the Vocabulary class."""

    def test_vocabulary_build_from_captions(self):
        """Test building a vocabulary from captions."""
        captions = ["the cat sits", "the dog runs", "a cat sleeps"]
        vocab = Vocabulary.build(captions, min_frequency=1, max_size=100)
        assert len(vocab.id_to_token) > 0
        assert vocab.PAD in vocab.token_to_id
        assert vocab.UNK in vocab.token_to_id
        assert vocab.BOS in vocab.token_to_id
        assert vocab.EOS in vocab.token_to_id

    def test_vocabulary_pad_id(self):
        """Test that pad_id is valid."""
        captions = ["test caption"]
        vocab = Vocabulary.build(captions)
        assert vocab.pad_id == vocab.token_to_id[vocab.PAD]
        assert vocab.pad_id == 0

    def test_vocabulary_encode(self):
        """Test encoding a caption."""
        captions = ["the cat sits on the mat"]
        vocab = Vocabulary.build(captions, max_size=100)
        max_length = 10
        encoded = vocab.encode("the cat", max_length)
        assert len(encoded) == max_length
        assert encoded[0] == vocab.bos_id
        assert encoded[-1] == vocab.pad_id

    def test_vocabulary_save_and_load(self, temp_dir):
        """Test saving and loading vocabulary."""
        captions = ["the cat sits", "the dog runs"]
        vocab = Vocabulary.build(captions)
        
        path = temp_dir / "vocab.json"
        vocab.save(path)
        
        loaded_vocab = Vocabulary.load(path)
        assert loaded_vocab.token_to_id == vocab.token_to_id
        assert loaded_vocab.id_to_token == vocab.id_to_token

    def test_vocabulary_min_frequency(self):
        """Test min_frequency filtering."""
        captions = ["cat"] * 5 + ["dog"] * 2 + ["bird"]
        vocab = Vocabulary.build(captions, min_frequency=2, max_size=100)
        # "cat" and "dog" should be in vocab, "bird" should not
        assert "cat" in vocab.token_to_id
        assert "dog" in vocab.token_to_id
        assert "bird" not in vocab.token_to_id


class TestReadCaptions:
    """Test the read_captions function."""

    def test_read_captions_basic(self, temp_dir):
        """Test reading captions from CSV."""
        csv_path = temp_dir / "captions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "Caption"])
            writer.writeheader()
            writer.writerow({"ID": "img_001", "Caption": "A cat sits"})
            writer.writerow({"ID": "img_002", "Caption": "A dog runs"})

        rows = read_captions(csv_path)
        assert len(rows) == 2
        assert rows[0] == ("img_001", "A cat sits")
        assert rows[1] == ("img_002", "A dog runs")

    def test_read_captions_limit(self, temp_dir):
        """Test reading limited number of captions."""
        csv_path = temp_dir / "captions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "Caption"])
            writer.writeheader()
            for i in range(10):
                writer.writerow({"ID": f"img_{i:03d}", "Caption": f"Caption {i}"})

        rows = read_captions(csv_path, limit=5)
        assert len(rows) == 5

    def test_read_captions_case_insensitive_headers(self, temp_dir):
        """Test that CSV headers are matched case-insensitively."""
        csv_path = temp_dir / "captions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "caption"])
            writer.writeheader()
            writer.writerow({"id": "img_001", "caption": "A cat"})

        rows = read_captions(csv_path)
        assert len(rows) == 1
        assert rows[0] == ("img_001", "A cat")


class TestRoCoCaptionDataset:
    """Test the RoCoCaptionDataset class."""

    def test_dataset_length(self, test_dataset_dir, sample_vocabulary):
        """Test that dataset reports correct length."""
        dataset = RoCoCaptionDataset(
            test_dataset_dir,
            "train",
            sample_vocabulary,
            image_size=224,
            max_length=16,
        )
        assert len(dataset) == 3

    def test_dataset_getitem(self, test_dataset_dir, sample_vocabulary):
        """Test getting an item from dataset."""
        dataset = RoCoCaptionDataset(
            test_dataset_dir,
            "train",
            sample_vocabulary,
            image_size=224,
            max_length=16,
        )
        image, caption, image_id = dataset[0]
        
        # Check image shape and dtype
        assert image.shape == (3, 224, 224)
        assert image.dtype == torch.float32
        assert image.min() >= -1.0 and image.max() <= 1.0
        
        # Check caption shape and dtype
        assert caption.shape == (16,)
        assert caption.dtype == torch.long
        
        # Check image_id is string
        assert isinstance(image_id, str)

    def test_dataset_normalization(self, test_dataset_dir, sample_vocabulary):
        """Test that images are properly normalized."""
        dataset = RoCoCaptionDataset(
            test_dataset_dir,
            "train",
            sample_vocabulary,
            image_size=224,
            max_length=16,
        )
        image, _, _ = dataset[0]
        
        # After normalization, image should be in [-1, 1] range
        assert -1.5 <= image.min() <= 1.5
        assert -1.5 <= image.max() <= 1.5

    def test_dataset_padding(self, test_dataset_dir, sample_vocabulary):
        """Test that captions are properly padded."""
        dataset = RoCoCaptionDataset(
            test_dataset_dir,
            "train",
            sample_vocabulary,
            image_size=224,
            max_length=20,
        )
        _, caption, _ = dataset[0]
        
        # First token should be BOS
        assert caption[0].item() == sample_vocabulary.bos_id
        
        # Last tokens should be padded (or EOS)
        pad_id = sample_vocabulary.pad_id
        # Find padding at the end
        for idx in range(len(caption) - 1, -1, -1):
            if caption[idx].item() != pad_id:
                break
        else:
            idx = -1
        
        # Everything after idx should be padding
        for i in range(idx + 1, len(caption)):
            assert caption[i].item() == pad_id
