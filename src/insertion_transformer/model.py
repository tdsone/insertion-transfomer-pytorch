"""Insertion Transformer model architecture."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PAD, EOS, Config, default_config


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with BIDIRECTIONAL attention (no causal mask).

    Unlike GPT which uses causal masking, the Insertion Transformer uses
    bidirectional attention since we need to see all tokens to decide
    where to insert.
    """

    def __init__(self, n_embd: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] input tensor
            mask: [B, T] boolean mask, True for valid positions, False for padding
        Returns:
            [B, T, C] output tensor
        """
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # [B, T, 3*C]
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, n_heads, T, T]

        # Apply padding mask if provided
        if mask is not None:
            # mask: [B, T] -> [B, 1, 1, T] for broadcasting
            attn_mask = mask[:, None, None, :]  # attend TO these positions
            attn = attn.masked_fill(~attn_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = attn @ v  # [B, n_heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, C)  # [B, T, C]
        out = self.proj(out)

        return out


class MLP(nn.Module):
    """Simple feed-forward network applied position-wise."""

    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, n_embd: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_heads, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class InsertionTransformer(nn.Module):
    """
    Decoder-only Insertion Transformer for language modeling.

    Key differences from standard GPT:
    1. Bidirectional attention (no causal mask) - we see all tokens to decide where to insert
    2. Outputs logits for (position, token) pairs
    3. Prepends a special "slot" token to allow insertion at position 0

    Architecture:
    - Input: partial sequence [t1, t2, ..., tn]
    - Prepend slot: [SLOT, t1, t2, ..., tn] -> n+1 positions
    - Each position i outputs:
        - position_logit: how likely to insert HERE (before token i)
        - token_logits: what token to insert

    Output interpretation:
    - Position i in output corresponds to inserting BEFORE position i in input
    - Position 0 = insert at the beginning
    - Position n = insert at the end (after last token)
    """

    def __init__(
        self,
        vocab_size: int,
        config: Config = default_config,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = config.n_embd
        self.max_seq_len = config.block_size + 1

        # Token embedding (includes special tokens: PAD=0, BOS=1, EOS=2)
        self.tok_emb = nn.Embedding(vocab_size, config.n_embd)

        # Learnable "slot" embedding prepended to represent insertion at position 0
        self.slot_emb = nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.02)

        # Positional embedding
        self.pos_emb = nn.Embedding(self.max_seq_len, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.n_embd, config.n_heads, config.dropout)
                for _ in range(config.n_layers)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output heads:
        # - token_head: predicts which token to insert [n_embd -> vocab_size]
        # - position_head: predicts insertion position weight [n_embd -> 1]
        self.token_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.position_head = nn.Linear(config.n_embd, 1, bias=False)

        # Weight tying: share token embeddings with output
        self.token_head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        hypo: torch.Tensor,  # [B, T] token indices
        hypo_len: Optional[torch.Tensor] = None,  # [B] actual lengths
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            hypo: [B, T] partial sequence (padded with PAD token)
            hypo_len: [B] actual lengths (optional, inferred if not provided)

        Returns:
            dict with:
                - 'position_logits': [B, T+1] logits for each insertion position
                - 'token_logits': [B, T+1, vocab_size] logits for each token at each position
                - 'insert_logp': [B, T+1, vocab_size] log P(insert token t at position p)
                - 'finish_logp': [B] log P(finish/EOS)
        """
        B, T = hypo.shape
        device = hypo.device

        # Infer lengths from padding if not provided
        if hypo_len is None:
            hypo_len = (hypo != PAD).sum(dim=1)

        # Create attention mask [B, T+1] - True for valid positions (including slot)
        # The slot (position 0) is always valid
        positions = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        token_mask = positions < hypo_len.unsqueeze(1)  # [B, T]
        # Prepend True for slot position
        mask = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=device), token_mask], dim=1
        )  # [B, T+1]

        # Token embeddings
        tok_emb = self.tok_emb(hypo)  # [B, T, n_embd]

        # Prepend slot embedding
        slot = self.slot_emb.expand(B, -1, -1)  # [B, 1, n_embd]
        x = torch.cat([slot, tok_emb], dim=1)  # [B, T+1, n_embd]

        # Add positional embeddings
        pos = torch.arange(T + 1, device=device)
        x = x + self.pos_emb(pos)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)  # [B, T+1, n_embd]

        # Compute logits
        position_logits = self.position_head(x).squeeze(-1)  # [B, T+1]
        token_logits = self.token_head(x)  # [B, T+1, vocab_size]

        # Mask invalid positions (after sequence end + 1 for final insertion point)
        # Valid positions: 0 to hypo_len (inclusive), so T+1 positions total for full sequence
        pos_mask = torch.arange(T + 1, device=device).unsqueeze(
            0
        ) <= hypo_len.unsqueeze(1)  # [B, T+1]
        position_logits = position_logits.masked_fill(~pos_mask, float("-inf"))

        # Compute log probabilities
        # P(insert at pos p) comes from position_logits (softmax over positions)
        # P(insert token t | pos p) comes from token_logits (softmax over vocab)
        position_logp = F.log_softmax(position_logits, dim=-1)  # [B, T+1]
        token_logp = F.log_softmax(token_logits, dim=-1)  # [B, T+1, vocab_size]

        # Combined: log P(insert token t at position p) = log P(pos) + log P(token|pos)
        insert_logp = position_logp.unsqueeze(-1) + token_logp  # [B, T+1, vocab_size]

        # Finish probability: inserting EOS at the last valid position
        # We define finish_logp as the logsumexp of inserting EOS at any position
        eos_logp = insert_logp[
            :, :, EOS
        ]  # [B, T+1] - log prob of inserting EOS at each position
        finish_logp = torch.logsumexp(
            eos_logp.masked_fill(~pos_mask, float("-inf")), dim=1
        )  # [B]

        return {
            "position_logits": position_logits,
            "token_logits": token_logits,
            "insert_logp": insert_logp,
            "finish_logp": finish_logp,
        }

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
