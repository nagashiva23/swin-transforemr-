"""Tests for losses module."""

from __future__ import annotations

import pytest
import torch

from src.losses import BiomedicalTermAwareFocalLoss


class TestBiomedicalTermAwareFocalLoss:
    """Test the BiomedicalTermAwareFocalLoss."""

    def test_loss_computation(self):
        """Test basic loss computation."""
        vocab_size = 100
        batch_size = 4
        seq_length = 16
        num_patches = 196
        
        word_weights = torch.ones(vocab_size)
        loss_fn = BiomedicalTermAwareFocalLoss(word_weights, gamma=2.0, pad_id=0)
        
        logits = torch.randn(batch_size, seq_length, vocab_size)
        targets = torch.randint(1, vocab_size, (batch_size, seq_length))
        attention = torch.randn(batch_size, seq_length, num_patches)
        
        loss, details = loss_fn(logits, targets, attention)
        
        assert loss.item() > 0
        assert "caption" in details
        assert "coverage" in details
        assert isinstance(loss, torch.Tensor)

    def test_loss_is_scalar(self):
        """Test that loss is a scalar."""
        word_weights = torch.ones(50)
        loss_fn = BiomedicalTermAwareFocalLoss(word_weights)
        
        logits = torch.randn(2, 10, 50)
        targets = torch.randint(0, 50, (2, 10))
        attention = torch.randn(2, 10, 100)
        
        loss, _ = loss_fn(logits, targets, attention)
        assert loss.shape == torch.Size([])

    def test_loss_respects_padding(self):
        """Test that padding tokens don't contribute to loss."""
        vocab_size = 50
        pad_id = 0
        word_weights = torch.ones(vocab_size)
        loss_fn = BiomedicalTermAwareFocalLoss(word_weights, pad_id=pad_id)
        
        logits = torch.randn(2, 10, vocab_size)
        # Make targets all padding
        targets = torch.zeros(2, 10, dtype=torch.long)
        attention = torch.randn(2, 10, 100)
        
        loss, details = loss_fn(logits, targets, attention)
        
        # Caption loss should be 0 since all tokens are padding
        assert details["caption"] == 0.0

    def test_focal_loss_emphasizes_hard_examples(self):
        """Test that focal loss emphasizes hard examples."""
        vocab_size = 50
        word_weights = torch.ones(vocab_size)
        
        # Create two losses: one with high gamma (more focus on hard examples)
        loss_fn_gamma2 = BiomedicalTermAwareFocalLoss(word_weights, gamma=2.0, pad_id=0)
        loss_fn_gamma0 = BiomedicalTermAwareFocalLoss(word_weights, gamma=0.0, pad_id=0)
        
        logits = torch.randn(2, 10, vocab_size)
        targets = torch.randint(1, vocab_size, (2, 10))
        attention = torch.randn(2, 10, 100)
        
        loss_gamma2, _ = loss_fn_gamma2(logits, targets, attention)
        loss_gamma0, _ = loss_fn_gamma0(logits, targets, attention)
        
        # Both should be positive
        assert loss_gamma2.item() > 0
        assert loss_gamma0.item() > 0

    def test_coverage_loss_term(self):
        """Test that coverage loss is computed."""
        vocab_size = 50
        word_weights = torch.ones(vocab_size)
        loss_fn = BiomedicalTermAwareFocalLoss(
            word_weights, 
            coverage_weight=0.1, 
            coverage_threshold=1.0,
            pad_id=0
        )
        
        logits = torch.randn(2, 10, vocab_size)
        targets = torch.randint(1, vocab_size, (2, 10))
        # Create attention with high values to trigger coverage loss
        attention = torch.ones(2, 10, 100) * 2.0
        
        loss, details = loss_fn(logits, targets, attention)
        
        assert details["coverage"] > 0.0

    def test_word_weights_effect(self):
        """Test that word weights affect the loss."""
        vocab_size = 50
        batch_size = 2
        seq_length = 10
        
        # Create weights where word 10 has high weight
        word_weights_uniform = torch.ones(vocab_size)
        word_weights_high = torch.ones(vocab_size)
        word_weights_high[10] = 5.0
        
        loss_fn_uniform = BiomedicalTermAwareFocalLoss(word_weights_uniform, pad_id=0)
        loss_fn_high = BiomedicalTermAwareFocalLoss(word_weights_high, pad_id=0)
        
        # Create data with word 10 in targets
        logits = torch.randn(batch_size, seq_length, vocab_size)
        targets = torch.ones(batch_size, seq_length, dtype=torch.long) * 10
        attention = torch.randn(batch_size, seq_length, 100)
        
        loss_uniform, _ = loss_fn_uniform(logits, targets, attention)
        loss_high, _ = loss_fn_high(logits, targets, attention)
        
        # Loss with higher weights should be higher
        assert loss_high.item() >= loss_uniform.item()

    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        vocab_size = 50
        word_weights = torch.ones(vocab_size)
        loss_fn = BiomedicalTermAwareFocalLoss(word_weights)
        
        logits = torch.randn(2, 10, vocab_size, requires_grad=True)
        targets = torch.randint(1, vocab_size, (2, 10))
        attention = torch.randn(2, 10, 100, requires_grad=True)
        
        loss, _ = loss_fn(logits, targets, attention)
        loss.backward()
        
        assert logits.grad is not None
        assert attention.grad is not None
        assert torch.any(logits.grad != 0)

    def test_large_batch(self):
        """Test loss computation with large batch."""
        vocab_size = 1000
        batch_size = 64
        seq_length = 32
        
        word_weights = torch.ones(vocab_size)
        loss_fn = BiomedicalTermAwareFocalLoss(word_weights)
        
        logits = torch.randn(batch_size, seq_length, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_length))
        attention = torch.randn(batch_size, seq_length, 196)
        
        loss, details = loss_fn(logits, targets, attention)
        
        assert not torch.isnan(loss)
        assert loss.item() > 0
