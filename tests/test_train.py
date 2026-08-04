"""Integration tests for training."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from src.data import RoCoCaptionDataset
from src.losses import BiomedicalTermAwareFocalLoss
from src.swin_transformer import SwinCaptioner


class TestTrainingIntegration:
    """Integration tests for the training pipeline."""

    def test_training_step(self, test_dataset_dir, sample_vocabulary, device):
        """Test a single training step."""
        # Setup model
        vocab_size = len(sample_vocabulary.id_to_token)
        model = SwinCaptioner(
            vocab_size=vocab_size,
            visual_dim=64,
            embedding_dim=64,
            hidden_dim=128,
            pad_id=sample_vocabulary.pad_id,
            image_size=224,
        )
        model.to(device)
        
        # Setup loss and optimizer
        word_weights = torch.ones(vocab_size)
        criterion = BiomedicalTermAwareFocalLoss(word_weights).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Setup dataset
        dataset = RoCoCaptionDataset(
            test_dataset_dir,
            "train",
            sample_vocabulary,
            image_size=224,
            max_length=16,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Run one training step
        model.train()
        for images, captions, _ in loader:
            images = images.to(device)
            captions = captions.to(device)
            
            logits, attention, _ = model(images, captions)
            loss, _ = criterion(logits, captions[:, 1:], attention)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            assert not torch.isnan(loss)
            assert loss.item() > 0
            break

    def test_validation_step(self, test_dataset_dir, sample_vocabulary, device):
        """Test a validation step."""
        vocab_size = len(sample_vocabulary.id_to_token)
        model = SwinCaptioner(
            vocab_size=vocab_size,
            visual_dim=64,
            embedding_dim=64,
            hidden_dim=128,
            pad_id=sample_vocabulary.pad_id,
            image_size=224,
        )
        model.to(device)
        
        word_weights = torch.ones(vocab_size)
        criterion = BiomedicalTermAwareFocalLoss(word_weights).to(device)
        
        dataset = RoCoCaptionDataset(
            test_dataset_dir,
            "valid",
            sample_vocabulary,
            image_size=224,
            max_length=16,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, captions, _ in loader:
                images = images.to(device)
                captions = captions.to(device)
                
                logits, attention, _ = model(images, captions)
                loss, _ = criterion(logits, captions[:, 1:], attention)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        assert avg_loss > 0
        assert not torch.isnan(torch.tensor(avg_loss))

    def test_gradient_accumulation(self, test_dataset_dir, sample_vocabulary, device):
        """Test gradient accumulation over multiple steps."""
        vocab_size = len(sample_vocabulary.id_to_token)
        model = SwinCaptioner(
            vocab_size=vocab_size,
            visual_dim=64,
            embedding_dim=64,
            hidden_dim=128,
            pad_id=sample_vocabulary.pad_id,
            image_size=224,
        )
        model.to(device)
        
        word_weights = torch.ones(vocab_size)
        criterion = BiomedicalTermAwareFocalLoss(word_weights).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        dataset = RoCoCaptionDataset(
            test_dataset_dir,
            "train",
            sample_vocabulary,
            image_size=224,
            max_length=16,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        model.train()
        
        # Accumulate gradients over multiple steps
        accumulation_steps = 3
        for step, (images, captions, _) in enumerate(loader):
            if step >= accumulation_steps:
                break
            
            images = images.to(device)
            captions = captions.to(device)
            
            logits, attention, _ = model(images, captions)
            loss, _ = criterion(logits, captions[:, 1:], attention)
            
            loss.backward()
            
            # Check that gradients are accumulating
            assert model.encoder.branches[0].pos_embed.grad is not None
        
        # Step the optimizer
        optimizer.step()
        
        # Gradients should be cleared after step
        optimizer.zero_grad()
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is None or torch.all(param.grad == 0)

    def test_checkpoint_save_load(self, test_dataset_dir, sample_vocabulary, temp_dir, device):
        """Test saving and loading checkpoints."""
        vocab_size = len(sample_vocabulary.id_to_token)
        model = SwinCaptioner(
            vocab_size=vocab_size,
            visual_dim=64,
            embedding_dim=64,
            hidden_dim=128,
            pad_id=sample_vocabulary.pad_id,
            image_size=224,
        )
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Run one step
        dataset = RoCoCaptionDataset(
            test_dataset_dir,
            "train",
            sample_vocabulary,
            image_size=224,
            max_length=16,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        word_weights = torch.ones(vocab_size)
        criterion = BiomedicalTermAwareFocalLoss(word_weights).to(device)
        
        model.train()
        for images, captions, _ in loader:
            images = images.to(device)
            captions = captions.to(device)
            
            logits, attention, _ = model(images, captions)
            loss, _ = criterion(logits, captions[:, 1:], attention)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break
        
        # Save checkpoint
        checkpoint_path = temp_dir / "checkpoint.pt"
        checkpoint = {
            "epoch": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        model2 = SwinCaptioner(
            vocab_size=vocab_size,
            visual_dim=64,
            embedding_dim=64,
            hidden_dim=128,
            pad_id=sample_vocabulary.pad_id,
            image_size=224,
        )
        model2.to(device)
        
        loaded = torch.load(checkpoint_path, map_location=device)
        model2.load_state_dict(loaded["model_state"])
        
        # Check that models have same weights
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_mixed_precision_compatibility(self, test_dataset_dir, sample_vocabulary, device):
        """Test that model works with mixed precision (if using CUDA)."""
        if device.type != "cuda":
            pytest.skip("Mixed precision test requires CUDA")
        
        vocab_size = len(sample_vocabulary.id_to_token)
        model = SwinCaptioner(
            vocab_size=vocab_size,
            visual_dim=64,
            embedding_dim=64,
            hidden_dim=128,
            pad_id=sample_vocabulary.pad_id,
            image_size=224,
        )
        model.to(device)
        
        dataset = RoCoCaptionDataset(
            test_dataset_dir,
            "train",
            sample_vocabulary,
            image_size=224,
            max_length=16,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        model.eval()
        with torch.cuda.amp.autocast():
            for images, captions, _ in loader:
                images = images.to(device)
                captions = captions.to(device)
                
                logits, attention, _ = model(images, captions)
                assert logits.dtype == torch.float16
                break
