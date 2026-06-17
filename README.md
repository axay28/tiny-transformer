# Tiny Transformer

A compact GPT-style language model built from scratch in PyTorch. This repo is designed to show the fundamentals recruiters actually care about: clean architecture, readable math, reproducible training, tests, and an end-to-end demo path from raw text to generated tokens.

## What Makes This Worth Looking At

- Implements a decoder-only Transformer without Hugging Face or high-level training frameworks.
- Includes causal self-attention, multi-head attention, residual blocks, layer norm, embeddings, generation, and checkpointing.
- Ships with a tiny character tokenizer so the model can train on any plain-text file.
- Keeps the code small enough to understand in one sitting, but structured like production Python.
- Includes smoke tests for masking, shapes, tokenization, and generation behavior.

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

Generate text from a checkpoint:

```bash
tiny-transformer generate --checkpoint runs/tiny-transformer.pt --prompt "To be" --max-new-tokens 160
```

Run tests:

```bash
pytest
```

## Project Layout

```text
src/tiny_transformer/
  cli.py          Command line interface for training and generation
  config.py       Model and training configuration
  data.py         Text dataset and batching utilities
  model.py        GPT-style Transformer implementation
  tokenizer.py    Character-level tokenizer
  train.py        Training loop, evaluation, checkpointing
tests/            Unit and smoke tests
data/             Tiny sample corpus
```

## Architecture

The model is intentionally small, but it follows the same structure as larger decoder-only LLMs:

1. Token and positional embeddings convert IDs into vectors.
2. Each Transformer block applies pre-norm causal self-attention.
3. Feed-forward layers expand and compress the hidden dimension.
4. Residual connections preserve gradient flow.
5. A tied-size language modeling head predicts the next token.

The attention mask is causal, so each position can only attend to itself and previous positions.

```mermaid
flowchart LR
    A["Raw text corpus"] --> B["Character tokenizer"]
    B --> C["Token IDs"]
    C --> D["Contiguous train/val batches"]
    D --> E["Token + position embeddings"]
    E --> F["Transformer block x N"]
    F --> G["Final layer norm"]
    G --> H["Language modeling head"]
    H --> I["Next-token logits"]
    I --> J["Cross-entropy loss during training"]
    I --> K["Top-k sampling during generation"]

    subgraph Block["Transformer block"]
        L["LayerNorm"] --> M["Causal multi-head self-attention"]
        M --> N["Residual add"]
        N --> O["LayerNorm"]
        O --> P["Feed-forward MLP"]
        P --> Q["Residual add"]
    end
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

## Portfolio Talking Points

- Why pre-layer-norm improves optimization stability.
- How causal masking prevents label leakage during next-token prediction.
- The tradeoff between character tokenization simplicity and sequence length.
- Why batching contiguous chunks is a good minimal language-modeling baseline.
- How temperature and top-k sampling change generation quality.

## Roadmap

- Add byte-pair encoding as an optional tokenizer.
- Add a web playground for interactive generation.
- Add attention heatmap visualization.
- Add mixed precision and gradient accumulation for larger local runs.
