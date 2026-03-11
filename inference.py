# Inference script for GPT model using MLX framework.
# This code loads a trained GPT model, encodes a prompt using the tiktoken tokenizer

import mlx.core as mx
import tiktoken
from model import GPT, GPTConfig

# Load Model
config = GPTConfig()
model = GPT(config)
model.load_weights("model_ckpt.safetensors")
enc = tiktoken.get_encoding("gpt2")

def generate(prompt, max_tokens=100, temp=0.8):
    tokens = mx.array(enc.encode(prompt))[None]
    
    for _ in range(max_tokens):
        logits = model(tokens)[:, -1] / temp
        next_token = mx.random.categorical(logits)
        tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)
        mx.eval(tokens)
    
    return enc.decode(tokens[0].tolist())

print("\n--- Testing Storytelling ---")
print(generate("Once upon a time, there was a little bird"))