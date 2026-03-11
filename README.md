# TinyStories Model Training

## Overview
This project explores whether a small AI model (58M parameters) can learn coherent English grammar and storytelling when trained on a high-quality synthetic dataset.

- **Model Size:** 58M parameters
- **Dataset:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (2.12M training examples)

## Hardware & Setup
- **Optimized for:** Apple M4 GPU
- **Backend:** MPS (Metal Performance Shaders)

## Model Architecture
- **Type:** GPT-style Transformer
- **Layers:** 6
- **Attention Heads:** 6

---

## Technical Implementation
To train efficiently on 16GB–18GB RAM hardware, we use:

| Technique               | Purpose                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| Memory Mapping          | Streams dataset from disk (`np.memmap`) to avoid RAM overload.         |
| Mixed Precision         | Uses `torch.autocast` (float16) for faster matrix multiplications.    |
| Gradient Accumulation   | Simulates larger batch sizes without exceeding VRAM limits.             |
| Checkpointing           | Saves the "Best Model" based on validation loss to prevent data loss.  |

---

## Training Progress
- **Initial Loss:** 11.0477
- **Validation Loss (1,000 iterations):** ~4.03

### Training Parameters
- **Learning Rate:** 6e-4 (Cosine Decay)
- **Batch Size:** 16 (optimized for thermal stability and multitasking)
- **Context Window:** 512 tokens

---
