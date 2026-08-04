"""Dataset utilities for the multi-scale Swin captioning pipeline."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\s\w]", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower().strip())


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    document_frequency: dict[str, int]

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    @classmethod
    def build(cls, captions: Iterable[str], min_frequency: int = 2, max_size: int = 12000) -> "Vocabulary":
        frequency: Counter[str] = Counter()
        document_frequency: Counter[str] = Counter()
        for caption in captions:
            tokens = tokenize(caption)
            frequency.update(tokens)
            document_frequency.update(set(tokens))
        specials = [cls.PAD, cls.UNK, cls.BOS, cls.EOS]
        learned = [token for token, count in frequency.most_common() if count >= min_frequency and token not in specials][: max_size - len(specials)]
        id_to_token = specials + learned
        return cls(
            token_to_id={token: index for index, token in enumerate(id_to_token)},
            id_to_token=id_to_token,
            document_frequency=dict(document_frequency),
        )

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    def encode(self, caption: str, max_length: int) -> list[int]:
        tokens = tokenize(caption)[: max_length - 2]
        ids = [self.bos_id] + [self.token_to_id.get(token, self.token_to_id[self.UNK]) for token in tokens] + [self.eos_id]
        return ids + [self.pad_id] * (max_length - len(ids))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"id_to_token": self.id_to_token, "document_frequency": self.document_frequency}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        payload = json.loads(path.read_text(encoding="utf-8"))
        id_to_token = payload["id_to_token"]
        return cls({token: index for index, token in enumerate(id_to_token)}, id_to_token, payload["document_frequency"])


def read_captions(csv_path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        
        # Find key headers case-insensitively
        id_key, caption_key = None, None
        if reader.fieldnames:
            for field in reader.fieldnames:
                field_lower = field.lower()
                if field_lower == "id":
                    id_key = field
                elif field_lower == "caption":
                    caption_key = field
        
        id_key = id_key or "ID"
        caption_key = caption_key or "Caption"

        for row in reader:
            val_id = row.get(id_key)
            val_caption = row.get(caption_key)
            if val_id and val_caption:
                rows.append((val_id, val_caption))
            if limit is not None and len(rows) >= limit:
                break
    return rows


class RoCoCaptionDataset(Dataset):
    def __init__(self, data_root: str | Path, split: str, vocabulary: Vocabulary, image_size: int = 224, max_length: int = 64, limit: int | None = None):
        self.data_root = Path(data_root)
        self.split = split
        self.vocabulary = vocabulary
        self.image_size = image_size
        self.max_length = max_length
        self.rows = read_captions(self.data_root / f"{split}_captions.csv", limit)
        img_dir = self.data_root / f"{split}_images"
        self.image_dir = img_dir / split if (img_dir / split).is_dir() else img_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        image_id, caption = self.rows[index]
        image_path = self.image_dir / f"{image_id}.jpg"
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((self.image_size, self.image_size))
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        pixels = (pixels - 0.5) / 0.5
        return pixels, torch.tensor(self.vocabulary.encode(caption, self.max_length), dtype=torch.long), image_id
