---
title: Tiny Transformer
sdk: gradio
app_file: app.py
pinned: false
---

# Tiny Transformer

A compact GPT-style language model built from scratch in PyTorch. The project includes clean architecture, readable math, reproducible training, tests, and an end-to-end demo path from raw text to generated tokens.

## Hosted Playground

[Try the Gradio demo on Hugging Face Spaces](https://huggingface.co/spaces/axay28/tiny-transformer).

![Tiny Transformer Hugging Face Space showing prompt, generation controls, and model output](docs/assets/huggingface-space.png)

## Features

- Implements a decoder-only Transformer without Hugging Face or high-level training frameworks.
- Includes causal self-attention, multi-head attention, residual blocks, layer norm, embeddings, generation, and checkpointing.
- Ships with character and byte-pair encoding tokenizers so the model can train on any plain-text file.
- Keeps the code small enough to understand in one sitting, but structured like production Python.
- Includes smoke tests for masking, shapes, tokenization, attention export, and generation behavior.
- Uses tied token/output weights and KV-cached autoregressive generation.
- Supports resumable optimizer checkpoints and validation perplexity evaluation.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Train on the included sample text:

```bash
tiny-transformer train --data data/tiny_shakespeare_excerpt.txt --steps 300 --device cpu
```

Use the optional BPE tokenizer, gradient accumulation, and mixed precision when you want a stronger local run:

```bash
tiny-transformer train \
  --data data/tiny_shakespeare_excerpt.txt \
  --tokenizer bpe \
  --bpe-vocab-size 128 \
  --grad-accum-steps 4 \
  --amp \
  --device mps
```

Generate text from a checkpoint:

```bash
tiny-transformer generate --checkpoint runs/tiny-transformer.pt --prompt "To be" --max-new-tokens 160
```

Resume a run to a new total step count:

```bash
tiny-transformer train \
  --data data/tiny_shakespeare_excerpt.txt \
  --output runs/tiny-transformer.pt \
  --steps 600 \
  --resume runs/tiny-transformer.pt
```

Evaluate loss, perplexity, and parameter count:

```bash
tiny-transformer evaluate \
  --checkpoint runs/tiny-transformer.pt \
  --data data/tiny_shakespeare_excerpt.txt
```

Export an attention heatmap:

```bash
tiny-transformer attention --checkpoint runs/tiny-transformer.pt --prompt "To be" --output runs/attention.svg
```

Launch the local playground:

```bash
tiny-transformer serve --checkpoint runs/tiny-transformer.pt
```

Deploy the hosted playground:

```bash
python -m pip install huggingface_hub
hf auth login
git remote add space https://huggingface.co/spaces/axay28/tiny-transformer
git push space main
```

The GitHub repository and Hugging Face Space are separate Git remotes. Push the same `main`
commit to both remotes so the hosted playground runs the documented implementation.

Run tests:

```bash
pytest
```

## Training Loss Artifact

The demo training run records train and validation loss at evaluation intervals. The committed example below was generated on CPU from the included sample corpus.

![Training loss curve](docs/assets/training-loss.svg)

Reproduce the CSV artifact:

```bash
tiny-transformer train \
  --data data/tiny_shakespeare_excerpt.txt \
  --output runs/loss-curve-demo.pt \
  --steps 80 \
  --batch-size 16 \
  --block-size 32 \
  --layers 2 \
  --heads 2 \
  --embedding 64 \
  --learning-rate 0.001 \
  --eval-interval 10 \
  --eval-batches 4 \
  --device cpu \
  --loss-history docs/assets/training-loss.csv
```

## Project Layout

```text
src/tiny_transformer/
  cli.py          Train, resume, generate, evaluate, serve, and inspect attention
  config.py       Model and training configuration
  data.py         Text dataset and batching utilities
  model.py        GPT-style Transformer implementation
  tokenizer.py    Character and byte-pair tokenizers
  train.py        Training, resume, perplexity evaluation, and checkpointing
  visualize.py    Attention heatmap export
  web.py          Local generation playground
tests/            Unit and smoke tests
data/             Tiny sample corpus
```

## Architecture

The model is intentionally small, but it follows the same structure as larger decoder-only LLMs:

1. Token and positional embeddings convert IDs into vectors.
2. Each Transformer block applies pre-norm causal self-attention.
3. Feed-forward layers expand and compress the hidden dimension.
4. Residual connections preserve gradient flow.
5. A weight-tied language modeling head predicts the next token.
6. Generation reuses per-layer key/value tensors until the context window rolls over.

The attention mask is causal, so each position can only attend to itself and previous positions.

```mermaid
flowchart LR
    A["Raw text corpus"] --> B["Char or BPE tokenizer"]
    B --> C["Token IDs"]
    C --> D["Contiguous train/val batches"]
    D --> E["Token + position embeddings"]
    E --> F1["LayerNorm"]
    F1 --> F2["Masked multi-head self-attention"]
    F2 --> F3["Residual add"]
    F3 --> F4["LayerNorm"]
    F4 --> F5["Feed-forward MLP"]
    F5 --> F6["Residual add"]
    F6 --> F7["Repeat for N layers"]
    F7 --> G["Final layer norm"]
    G --> H["Language modeling head"]
    H --> I["Next-token logits"]
    I --> J["Cross-entropy loss during training"]
    I --> K["KV-cached top-k generation"]
    F2 --> L["Attention heatmap export"]
    K --> M["Local web playground"]
```

## Example Configuration

The CLI defaults train quickly on CPU. For the included tiny corpus, the command uses a
32-token context window; for larger text files, 128 tokens is a good next step:

```python
ModelConfig(
    vocab_size=128,
    block_size=128,
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.1,
)
```

Increase `n_layer`, `n_head`, and `n_embd` for a stronger demo once the training loop is validated.

## Model Card

The hosted Gradio demo uses a deliberately tiny character-level checkpoint trained on the
sample corpus in `data/tiny_shakespeare_excerpt.txt`. The checkpoint is meant to demonstrate
the end-to-end Transformer pipeline: tokenization, causal masking, training, checkpoint
loading, and top-k sampling.

This model is useful for inspecting architecture and generation mechanics, not for factual
answers or broad language understanding. Its outputs are best with prompts similar to the
sample corpus, such as `To be`, `Attention`, or `The model`. Higher temperature settings can
produce noisy text because the model is intentionally small and trained on a compact dataset.
