#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
TSLANet Encoder adapted for OpenTSLM.

This module provides a TSLANet-based encoder that is compatible with the
OpenTSLM pipeline. It includes:
- Adaptive Spectral Block (ASB) for frequency-domain processing
- Interactive Convolutional Block (ICB) for local feature extraction
- get_embedding() method for RAG-ready sequence-level embeddings
"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

from model_config import ENCODER_OUTPUT_DIM, PATCH_SIZE
from model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class ICB(nn.Module):
    """Interactive Convolutional Block."""
    
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] -> transpose for conv1d
        x = x.transpose(1, 2)  # [B, D, N]
        
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)  # [B, N, D]
        return x


class PatchEmbed(nn.Module):
    """Patch Embedding with overlapping patches."""
    
    def __init__(self, seq_len: int, patch_size: int = 8, in_chans: int = 1, embed_dim: int = 128):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> [B, D, N] -> [B, N, D]
        x_out = self.proj(x).transpose(1, 2)
        return x_out


class AdaptiveSpectralBlock(nn.Module):
    """Adaptive Spectral Block for frequency-domain processing."""
    
    def __init__(self, dim: int, adaptive_filter: bool = True):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft: torch.Tensor) -> torch.Tensor:
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy and compute median
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)

        # Normalize energy
        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)

        # Create adaptive mask with straight-through estimator
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # Adaptive High Frequency Mask
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted = x_weighted + x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)

        return x


class TSLANetLayer(nn.Module):
    """Single TSLANet layer with ASB and ICB."""
    
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 3.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_icb: bool = True,
        use_asb: bool = True,
        adaptive_filter: bool = True,
    ):
        super().__init__()
        self.use_icb = use_icb
        self.use_asb = use_asb
        
        self.norm1 = nn.LayerNorm(dim)
        self.asb = AdaptiveSpectralBlock(dim, adaptive_filter=adaptive_filter)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_icb and self.use_asb:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.use_asb:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


class TSLANetEncoder(TimeSeriesEncoderBase):
    """
    TSLANet encoder adapted for OpenTSLM.
    
    Input: [B, L] - raw time series
    Output: [B, N, D] - patch-level features (compatible with TransformerCNNEncoder)
    
    Additional methods:
    - get_embedding(): Returns [B, D] for RAG retrieval
    - pretrain_forward(): Returns masked prediction and target for pretraining
    """
    
    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.15,
        patch_size: int = 8,
        depth: int = 2,
        mlp_ratio: float = 3.0,
        use_icb: bool = True,
        use_asb: bool = True,
        adaptive_filter: bool = True,
        max_seq_len: int = 4096,
    ):
        """
        Args:
            output_dim: Embedding dimension (default: 128)
            dropout: Dropout rate
            patch_size: Size of each patch
            depth: Number of TSLANet layers
            mlp_ratio: MLP hidden dimension ratio
            use_icb: Whether to use Interactive Convolutional Block
            use_asb: Whether to use Adaptive Spectral Block
            adaptive_filter: Whether to use adaptive high-frequency filter
            max_seq_len: Maximum sequence length supported
        """
        super().__init__(output_dim, dropout)
        
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.use_icb = use_icb
        self.use_asb = use_asb
        self.adaptive_filter = adaptive_filter
        
        # Patch embedding (will be reinitialized based on actual seq_len)
        self._seq_len = max_seq_len
        self.patch_embed = PatchEmbed(
            seq_len=max_seq_len, 
            patch_size=patch_size,
            in_chans=1, 
            embed_dim=output_dim
        )
        
        # Calculate max number of patches
        stride = patch_size // 2
        max_num_patches = int((max_seq_len - patch_size) / stride + 1)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_num_patches, output_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, dropout, depth)]
        
        # TSLANet layers
        self.tsla_blocks = nn.ModuleList([
            TSLANetLayer(
                dim=output_dim,
                mlp_ratio=mlp_ratio,
                drop=dropout,
                drop_path=dpr[i],
                use_icb=use_icb,
                use_asb=use_asb,
                adaptive_filter=adaptive_filter,
            )
            for i in range(depth)
        ])
        
        # Initialize weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape [B, L], raw time series.
        Returns:
            FloatTensor of shape [B, N, D], patch-level features.
        """
        B, L = x.shape
        
        # Reshape to [B, 1, L] for Conv1d
        x = x.unsqueeze(1)
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]
        N = x.size(1)
        
        # Add positional embedding (truncate if needed)
        if N > self.pos_embed.size(1):
            raise ValueError(
                f"Sequence produces {N} patches, but max is {self.pos_embed.size(1)}. "
                f"Increase max_seq_len or reduce sequence length."
            )
        x = x + self.pos_embed[:, :N, :]
        x = self.pos_drop(x)
        
        # Apply TSLANet layers
        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)
        
        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get sequence-level embedding for RAG retrieval.
        
        Args:
            x: FloatTensor of shape [B, L], raw time series.
        Returns:
            FloatTensor of shape [B, D], sequence embedding.
        """
        features = self.forward(x)  # [B, N, D]
        return features.mean(dim=1)  # [B, D] - global average pooling

    def pretrain_forward(
        self, 
        x: torch.Tensor, 
        mask_ratio: float = 0.4
    ) -> tuple:
        """
        Forward pass for pretraining with masked reconstruction.
        
        Args:
            x: FloatTensor of shape [B, L], raw time series.
            mask_ratio: Ratio of patches to mask.
        Returns:
            Tuple of (predictions, targets, mask)
        """
        B, L = x.shape
        x = x.unsqueeze(1)  # [B, 1, L]
        
        # Get patch embeddings
        x_patched = self.patch_embed(x)  # [B, N, D]
        N = x_patched.size(1)
        
        # Add positional embedding
        x = x_patched + self.pos_embed[:, :N, :]
        x = self.pos_drop(x)
        
        # Random masking
        x_masked, mask = self._random_masking(x, mask_ratio)
        
        # Apply TSLANet layers
        for tsla_blk in self.tsla_blocks:
            x_masked = tsla_blk(x_masked)
        
        return x_masked, x_patched, mask

    def _random_masking(self, x: torch.Tensor, mask_ratio: float) -> tuple:
        """
        Random masking for pretraining.
        
        Args:
            x: [B, N, D] patch embeddings
            mask_ratio: Ratio to mask
        Returns:
            Tuple of (masked_x, mask)
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Zero out removed patches
        x_removed = torch.zeros(B, N - len_keep, D, device=x.device, dtype=x.dtype)
        x_ = torch.cat([x_kept, x_removed], dim=1)
        
        # Restore order
        x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        # Create binary mask: 0 is keep, 1 is masked
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask


if __name__ == "__main__":
    # Test the encoder
    print("Testing TSLANetEncoder...")
    
    encoder = TSLANetEncoder(
        output_dim=128,
        patch_size=8,
        depth=2,
        max_seq_len=256,
    )
    
    # Test forward
    x = torch.randn(4, 140)  # ECG200 length
    out = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test get_embedding
    emb = encoder.get_embedding(x)
    print(f"Embedding shape: {emb.shape}")
    
    # Test pretrain_forward
    pred, target, mask = encoder.pretrain_forward(x, mask_ratio=0.4)
    print(f"Pretrain - pred: {pred.shape}, target: {target.shape}, mask: {mask.shape}")
    
    # Parameter count
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"Total parameters: {num_params:,}")
    
    print("\n✅ TSLANetEncoder test passed!")
