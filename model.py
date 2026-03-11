# GPT model implementation using MLX framework. 
# This code defines a GPTConfig dataclass for model configuration and a 
# GPT class that implements the GPT architecture using MLX's native modules. T
# he model includes token and position embeddings, multiple Transformer blocks, 
# layer normalization, and a linear layer for output. Weight tying is used between 
# the token embedding and the output layer.

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 512  # Context window
    n_layer: int = 12      # Depth
    n_head: int = 8        # Attention heads
    n_embd: int = 512      # Embedding dimension

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # MLX-native Transformer blocks
        self.blocks = [nn.TransformerEncoderLayer(config.n_embd, config.n_head) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight # Weight Tying

    def __call__(self, x):
        b, t = x.shape
        mask = nn.MultiHeadAttention.create_additive_causal_mask(t)
        x = self.wte(x) + self.wpe(mx.arange(t))
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.ln_f(x))