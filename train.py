"""Train the multi-scale gated Swin captioning pipeline."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import RoCoCaptionDataset, Vocabulary, read_captions
from src.losses import BiomedicalTermAwareFocalLoss
from src.swin_transformer import SwinCaptioner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Swin-inspired multi-scale captioning model.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/swin_pipeline"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-vocab", type=int, default=12000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def word_weights(vocabulary: Vocabulary, caption_count: int) -> torch.Tensor:
    weights = []
    for token in vocabulary.id_to_token:
        if token in [vocabulary.PAD, vocabulary.BOS, vocabulary.EOS, vocabulary.UNK]:
            weights.append(1.0)
        else:
            frequency = vocabulary.document_frequency.get(token, 0)
            weights.append(1.0 + 0.3 * math.log((caption_count + 1) / (frequency + 1)))
    tensor = torch.tensor(weights, dtype=torch.float32)
    tensor /= tensor.mean()
    tensor[vocabulary.pad_id] = 0.0
    return tensor.clamp(max=3.0)


def run_epoch(model, loader, criterion, optimizer, device, training: bool) -> float:
    model.train(training)
    total_loss, total_items = 0.0, 0
    progress = tqdm(loader, leave=False, desc="train" if training else "valid")
    for images, captions, _ in progress:
        images, captions = images.to(device), captions.to(device)
        with torch.set_grad_enabled(training):
            logits, attention, _ = model(images, captions)
            loss, details = criterion(logits, captions[:, 1:], attention)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_items += images.size(0)
        progress.set_postfix(loss=f"{loss.item():.3f}", caption=f"{details['caption']:.3f}")
    return total_loss / max(total_items, 1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_captions(args.data_root / "train_captions.csv", args.train_limit)
    vocabulary = Vocabulary.build((caption for _, caption in train_rows), max_size=args.max_vocab)
    vocabulary.save(args.output_dir / "vocabulary.json")
    train_set = RoCoCaptionDataset(args.data_root, "train", vocabulary, args.image_size, args.max_length, args.train_limit)
    valid_set = RoCoCaptionDataset(args.data_root, "valid", vocabulary, args.image_size, args.max_length, args.valid_limit)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    valid_loader = DataLoader(valid_set, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = SwinCaptioner(len(vocabulary.id_to_token), visual_dim=256, pad_id=vocabulary.pad_id, image_size=args.image_size).to(device)
    criterion = BiomedicalTermAwareFocalLoss(word_weights(vocabulary, len(train_rows))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    print(f"device={device}, train={len(train_set)}, valid={len(valid_set)}, vocab={len(vocabulary.id_to_token)}")
    best_valid = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, True)
        valid_loss = run_epoch(model, valid_loader, criterion, optimizer, device, False)
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")
        checkpoint = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "valid_loss": valid_loss, "vocab_size": len(vocabulary.id_to_token)}
        torch.save(checkpoint, args.output_dir / "last.pt")
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
