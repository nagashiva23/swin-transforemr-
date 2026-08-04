"""Multi-scale gated Swin-inspired image captioning model."""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.projection(images)
        features = features.flatten(2).transpose(1, 2)
        return self.norm(features)


class SwinBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        return x


class SwinStem(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int, depth: int = 2, heads: int = 4, image_size: int = 224):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([SwinBlock(embed_dim, heads=heads) for _ in range(depth)])
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.patch_embed(images)
        features = features + self.pos_embed
        for block in self.blocks:
            features = block(features)
        return features


class MultiScaleGatedSwinEncoder(nn.Module):
    def __init__(self, visual_dim: int = 256, depth: int = 2, heads: int = 4, image_size: int = 224):
        super().__init__()
        self.branches = nn.ModuleList([
            SwinStem(patch_size=4, embed_dim=visual_dim, depth=depth, heads=heads, image_size=image_size),
            SwinStem(patch_size=8, embed_dim=visual_dim, depth=depth, heads=heads, image_size=image_size),
            SwinStem(patch_size=16, embed_dim=visual_dim, depth=depth, heads=heads, image_size=image_size),
        ])
        self.gate = nn.Linear(visual_dim * 3, 3)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = [branch(images) for branch in self.branches]
        
        batch_size, L_0, embed_dim = features[0].shape
        H_0 = W_0 = int(math.sqrt(L_0))
        
        aligned_features = []
        for feature in features:
            L = feature.shape[1]
            if L == L_0:
                aligned_features.append(feature)
            else:
                H = W = int(math.sqrt(L))
                feat_2d = feature.transpose(1, 2).view(batch_size, embed_dim, H, W)
                feat_2d_aligned = F.interpolate(feat_2d, size=(H_0, W_0), mode="bilinear", align_corners=False)
                feat_aligned = feat_2d_aligned.flatten(2).transpose(1, 2)
                aligned_features.append(feat_aligned)
                
        descriptors = torch.cat([feature.mean(dim=1) for feature in features], dim=1)
        gate_weights = F.softmax(self.gate(descriptors), dim=1)
        
        fused = torch.zeros_like(features[0])
        for index, feature in enumerate(aligned_features):
            fused = fused + gate_weights[:, index][:, None, None] * feature
            
        return fused, gate_weights


class VisualEvidenceDecoder(nn.Module):
    def __init__(self, vocab_size: int, visual_dim: int = 256, embedding_dim: int = 256, hidden_dim: int = 512, pad_id: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.query = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.feature_projection = nn.Linear(visual_dim, hidden_dim, bias=False)
        self.attention_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_score = nn.Linear(hidden_dim, 1, bias=False)
        self.memory_gate = nn.Linear(embedding_dim + visual_dim + hidden_dim, visual_dim)
        self.candidate = nn.Linear(embedding_dim + visual_dim + visual_dim + hidden_dim, hidden_dim)
        self.state_gate = nn.Linear(embedding_dim + visual_dim + visual_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim + visual_dim + visual_dim, vocab_size)

    def step(self, previous_word: torch.Tensor, features: torch.Tensor, projected_features: torch.Tensor, hidden: torch.Tensor, memory: torch.Tensor):
        embedding = self.embedding(previous_word)
        query = torch.tanh(self.query(torch.cat([embedding, hidden], dim=1)))
        energy = self.attention_score(torch.tanh(projected_features + self.attention_projection(query).unsqueeze(1))).squeeze(-1)
        attention = F.softmax(energy, dim=1)
        context = torch.bmm(attention.unsqueeze(1), features).squeeze(1)

        decoder_input = torch.cat([embedding, context, hidden], dim=1)
        memory_gate = torch.sigmoid(self.memory_gate(decoder_input))
        memory = memory_gate * context + (1.0 - memory_gate) * memory

        state_input = torch.cat([embedding, context, memory, hidden], dim=1)
        candidate = torch.tanh(self.candidate(state_input))
        state_gate = torch.sigmoid(self.state_gate(state_input))
        hidden = state_gate * candidate + (1.0 - state_gate) * hidden
        logits = self.output(torch.cat([hidden, context, memory], dim=1))
        return logits, hidden, memory, attention

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.size(0)
        hidden = features.new_zeros(batch_size, self.hidden_dim)
        memory = features.new_zeros(batch_size, features.size(-1))
        projected_features = self.feature_projection(features)
        logits, attentions = [], []
        for step in range(captions.size(1) - 1):
            output, hidden, memory, attention = self.step(captions[:, step], features, projected_features, hidden, memory)
            logits.append(output)
            attentions.append(attention)
        return torch.stack(logits, dim=1), torch.stack(attentions, dim=1)


class SwinCaptioner(nn.Module):
    def __init__(self, vocab_size: int, visual_dim: int = 256, embedding_dim: int = 256, hidden_dim: int = 512, pad_id: int = 0, image_size: int = 224):
        super().__init__()
        self.encoder = MultiScaleGatedSwinEncoder(visual_dim=visual_dim, image_size=image_size)
        self.decoder = VisualEvidenceDecoder(vocab_size, visual_dim, embedding_dim, hidden_dim, pad_id)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, scale_weights = self.encoder(images)
        logits, attention = self.decoder(features, captions)
        return logits, attention, scale_weights
