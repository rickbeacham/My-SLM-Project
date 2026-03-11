import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import os
from functools import partial
from model import GPT, GPTConfig

# --- 1. Configuration & Hyperparameters ---
batch_size = 16
block_size = 512
max_iters = 5000
eval_interval = 500  # How often to check validation loss
eval_batches = 20    # Number of batches to use for validation estimate
learning_rate = 6e-4

# --- 2. Data Loading (Memory Mapped) ---
# Ensuring paths match your data/tinystories/ folder structure
train_path = 'data/tinystories/train.bin'
val_path = 'data/tinystories/val.bin'

if not os.path.exists(train_path) or not os.path.exists(val_path):
    raise FileNotFoundError("Binary data files not found. Please run data/tinystories/prepare.py first.")

train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')

def get_batch(split='train'):
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = mx.array(np.stack([data[i:i+block_size] for i in ix]).astype(np.int32))
    y = mx.array(np.stack([data[i+1:i+1+block_size] for i in ix]).astype(np.int32))
    return x, y

# --- 3. Initialization ---
config = GPTConfig()
model = GPT(config)
mx.eval(model.parameters()) # Initialize weights on M4 GPU

optimizer = optim.AdamW(learning_rate=learning_rate)

# IMPORTANT: Capture the state for the MLX compiler
state = [model.state, optimizer.state]

# --- 4. Training & Validation Logic ---
def loss_fn(model, x, y):
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# The 'partial' decorator ensures model and optimizer states are tracked correctly
@partial(mx.compile, inputs=state, outputs=state)
def step(x, y):
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    return loss

def estimate_loss():
    # Helper to calculate average loss on validation set
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(split='val')
        loss = loss_fn(model, x, y)
        losses.append(loss.item())
    return np.mean(losses)

# --- 5. Main Training Loop ---
best_val_loss = float('inf')
print(f"Starting training on M4 GPU. Model Size: ~58M Parameters")

for i in range(max_iters):
    # Training Step
    x, y = get_batch('train')
    loss = step(x, y)
    
    # Trigger M4 GPU work by evaluating the captured state and the loss
    mx.eval(state, loss) 
    
    # Logging
    if i % 100 == 0:
        print(f"Iter {i}: Train Loss {loss.item():.4f}")

    # Validation & Checkpointing
    if i % eval_interval == 0 or i == max_iters - 1:
        val_loss = estimate_loss()
        print(f"--- VALIDATION at Iter {i}: Loss {val_loss:.4f} ---")
        
        # Save weights every interval
        model.save_weights("model_ckpt.safetensors")
        
        # Save a dedicated 'best' file if validation improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights("best_model.safetensors")
            print(f">>> New Best Model Saved (Val Loss: {val_loss:.4f})")

print(f"Training Complete. Lowest Validation Loss: {best_val_loss:.4f}")