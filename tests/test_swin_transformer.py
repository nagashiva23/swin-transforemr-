"""Tests for swin_transformer module."""

from __future__ import annotations

import pytest
import torch

from src.swin_transformer import (
    ConvNormAct,
    PatchEmbedding,
    SwinBlock,
    SwinStem,
    MultiScaleGatedSwinEncoder,
    VisualEvidenceDecoder,
    SwinCaptioner,
)


class TestConvNormAct:
    """Test the ConvNormAct block."""

    def test_forward_pass(self):
        """Test forward pass through ConvNormAct."""
        layer = ConvNormAct(3, 64, stride=1)
        x = torch.randn(2, 3, 224, 224)
        output = layer(x)
        assert output.shape == (2, 64, 224, 224)

    def test_stride_downsampling(self):
        """Test that stride parameter downsamples."""
        layer = ConvNormAct(3, 64, stride=2)
        x = torch.randn(1, 3, 224, 224)
        output = layer(x)
        assert output.shape == (1, 64, 112, 112)


class TestPatchEmbedding:
    """Test the PatchEmbedding layer."""

    def test_patch_embedding_output_shape(self):
        """Test patch embedding output shape."""
        patch_size = 16
        embed_dim = 768
        layer = PatchEmbedding(patch_size, embed_dim)
        x = torch.randn(2, 3, 224, 224)
        output = layer(x)
        
        # Expected: (batch_size, num_patches, embed_dim)
        expected_patches = (224 // patch_size) ** 2
        assert output.shape == (2, expected_patches, embed_dim)

    def test_different_patch_sizes(self):
        """Test with different patch sizes."""
        embed_dim = 256
        for patch_size in [4, 8, 16]:
            layer = PatchEmbedding(patch_size, embed_dim)
            x = torch.randn(1, 3, 224, 224)
            output = layer(x)
            expected_patches = (224 // patch_size) ** 2
            assert output.shape == (1, expected_patches, embed_dim)


class TestSwinBlock:
    """Test the SwinBlock transformer block."""

    def test_swin_block_forward(self):
        """Test forward pass through SwinBlock."""
        dim = 256
        block = SwinBlock(dim, heads=4)
        x = torch.randn(2, 196, dim)  # 196 patches
        output = block(x)
        assert output.shape == x.shape

    def test_residual_connection(self):
        """Test that residual connection preserves shape."""
        dim = 256
        block = SwinBlock(dim, heads=4)
        x = torch.randn(1, 100, dim)
        output = block(x)
        assert output.shape == x.shape

    def test_dropout_effect(self):
        """Test that dropout is applied during training."""
        dim = 128
        block = SwinBlock(dim, heads=4, dropout=0.5)
        block.train()
        x = torch.randn(2, 50, dim)
        
        # Run multiple times and check outputs differ (due to dropout)
        out1 = block(x)
        out2 = block(x)
        assert not torch.allclose(out1, out2)


class TestSwinStem:
    """Test the SwinStem encoder."""

    def test_swin_stem_output_shape(self):
        """Test SwinStem output shape."""
        stem = SwinStem(patch_size=4, embed_dim=256, depth=2, heads=4, image_size=224)
        x = torch.randn(2, 3, 224, 224)
        output = stem(x)
        
        # Expected: (batch_size, num_patches, embed_dim)
        expected_patches = (224 // 4) ** 2
        assert output.shape == (2, expected_patches, 256)

    def test_positional_encoding(self):
        """Test that positional embeddings are applied."""
        stem = SwinStem(patch_size=16, embed_dim=256, depth=1, image_size=224)
        x = torch.randn(2, 3, 224, 224)
        
        # Get embedding before pos_embed is added
        with torch.no_grad():
            embedded = stem.patch_embed(x)
            output = embedded + stem.pos_embed
            
        assert output.shape == embedded.shape


class TestMultiScaleGatedSwinEncoder:
    """Test the MultiScaleGatedSwinEncoder."""

    def test_encoder_forward_pass(self):
        """Test forward pass through encoder."""
        encoder = MultiScaleGatedSwinEncoder(visual_dim=256, depth=2, heads=4, image_size=224)
        x = torch.randn(2, 3, 224, 224)
        features, gate_weights = encoder(x)
        
        # Check features shape
        expected_patches = (224 // 4) ** 2
        assert features.shape == (2, expected_patches, 256)
        
        # Check gate weights
        assert gate_weights.shape == (2, 3)
        assert torch.allclose(gate_weights.sum(dim=1), torch.ones(2))

    def test_gate_weights_sum_to_one(self):
        """Test that gate weights sum to 1."""
        encoder = MultiScaleGatedSwinEncoder(visual_dim=128, depth=1, heads=2, image_size=224)
        x = torch.randn(4, 3, 224, 224)
        _, gate_weights = encoder(x)
        
        assert gate_weights.shape[0] == 4
        assert gate_weights.shape[1] == 3
        assert torch.allclose(gate_weights.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_feature_alignment(self):
        """Test that features from different scales are properly aligned."""
        encoder = MultiScaleGatedSwinEncoder(visual_dim=256, depth=1, image_size=224)
        x = torch.randn(2, 3, 224, 224)
        features, _ = encoder(x)
        
        # All patch sets should be aligned to same size
        expected_patches = (224 // 4) ** 2
        assert features.shape[1] == expected_patches


class TestVisualEvidenceDecoder:
    """Test the VisualEvidenceDecoder."""

    def test_decoder_forward_pass(self):
        """Test forward pass through decoder."""
        vocab_size = 100
        visual_dim = 256
        embedding_dim = 256
        hidden_dim = 512
        
        decoder = VisualEvidenceDecoder(
            vocab_size, visual_dim, embedding_dim, hidden_dim, pad_id=0
        )
        
        features = torch.randn(2, 196, visual_dim)  # 196 patches
        captions = torch.randint(0, vocab_size, (2, 16))
        
        logits, attention = decoder(features, captions)
        
        # Check shapes
        assert logits.shape == (2, 15, vocab_size)  # seq_length - 1
        assert attention.shape == (2, 15, 196)

    def test_step_function(self):
        """Test the step function of decoder."""
        vocab_size = 50
        visual_dim = 128
        embedding_dim = 128
        hidden_dim = 256
        batch_size = 2
        
        decoder = VisualEvidenceDecoder(
            vocab_size, visual_dim, embedding_dim, hidden_dim, pad_id=0
        )
        
        features = torch.randn(batch_size, 100, visual_dim)
        previous_word = torch.randint(0, vocab_size, (batch_size,))
        hidden = torch.zeros(batch_size, hidden_dim)
        memory = torch.zeros(batch_size, visual_dim)
        
        projected_features = decoder.feature_projection(features)
        
        logits, new_hidden, new_memory, attention = decoder.step(
            previous_word, features, projected_features, hidden, memory
        )
        
        assert logits.shape == (batch_size, vocab_size)
        assert new_hidden.shape == hidden.shape
        assert new_memory.shape == memory.shape
        assert attention.shape == (batch_size, 100)


class TestSwinCaptioner:
    """Test the complete SwinCaptioner model."""

    def test_captioner_forward_pass(self, swin_captioner, sample_batch, device):
        """Test forward pass through the full captioner."""
        swin_captioner.to(device)
        images, captions = sample_batch
        images = images.to(device)
        captions = captions.to(device)
        
        logits, attention, scale_weights = swin_captioner(images, captions)
        
        # Check output shapes
        assert logits.shape == (2, 15, 10)  # (batch, seq_len-1, vocab_size)
        assert attention.shape[0] == 2
        assert attention.shape[1] == 15
        assert scale_weights.shape == (2, 3)

    def test_gradient_flow(self, swin_captioner, sample_batch, device):
        """Test that gradients flow through the model."""
        swin_captioner.to(device)
        images, captions = sample_batch
        images = images.to(device)
        captions = captions.to(device)
        
        logits, _, _ = swin_captioner(images, captions)
        loss = logits.mean()
        loss.backward()
        
        # Check that gradients are computed
        for name, param in swin_captioner.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_eval_mode(self, swin_captioner, sample_batch, device):
        """Test model in evaluation mode."""
        swin_captioner.eval()
        swin_captioner.to(device)
        images, captions = sample_batch
        images = images.to(device)
        captions = captions.to(device)
        
        with torch.no_grad():
            logits, attention, scale_weights = swin_captioner(images, captions)
        
        assert logits.shape == (2, 15, 10)

    def test_different_batch_sizes(self, swin_captioner, device):
        """Test model with different batch sizes."""
        swin_captioner.to(device)
        
        for batch_size in [1, 2, 4]:
            images = torch.randn(batch_size, 3, 224, 224, device=device)
            captions = torch.randint(0, 10, (batch_size, 16), device=device)
            
            logits, attention, scale_weights = swin_captioner(images, captions)
            
            assert logits.shape == (batch_size, 15, 10)
            assert scale_weights.shape == (batch_size, 3)
