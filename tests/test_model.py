import importlib.util

import pytest

torch_available = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch is not installed")

if torch_available:
    import torch

    from tiny_transformer.config import ModelConfig
    from tiny_transformer.model import TinyTransformer


def test_forward_shape_and_loss() -> None:
    config = ModelConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = TinyTransformer(config)
    x = torch.randint(0, config.vocab_size, (4, config.block_size))
    logits, loss = model(x, x)

    assert logits.shape == (4, config.block_size, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0


def test_generation_extends_sequence() -> None:
    config = ModelConfig(vocab_size=7, block_size=6, n_layer=1, n_head=1, n_embd=8, dropout=0.0)
    model = TinyTransformer(config)
    x = torch.zeros((1, 3), dtype=torch.long)

    out = model.generate(x, max_new_tokens=5, top_k=3)

    assert out.shape == (1, 8)
