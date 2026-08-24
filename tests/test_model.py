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


def test_cached_generation_matches_uncached_generation() -> None:
    config = ModelConfig(vocab_size=7, block_size=12, n_layer=2, n_head=2, n_embd=8, dropout=0.0)
    model = TinyTransformer(config).eval()
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)

    torch.manual_seed(7)
    cached = model.generate(x, max_new_tokens=5, top_k=3, use_cache=True)
    torch.manual_seed(7)
    uncached = model.generate(x, max_new_tokens=5, top_k=3, use_cache=False)

    assert torch.equal(cached, uncached)


def test_embedding_and_output_weights_are_tied() -> None:
    model = TinyTransformer(ModelConfig(vocab_size=7, n_layer=1, n_head=1, n_embd=8))

    assert model.lm_head.weight is model.token_embedding.weight


def test_attention_maps_have_layer_head_and_causal_shape() -> None:
    config = ModelConfig(vocab_size=7, block_size=6, n_layer=2, n_head=2, n_embd=8, dropout=0.0)
    model = TinyTransformer(config)
    x = torch.zeros((1, 4), dtype=torch.long)

    maps = model.attention_maps(x)

    assert len(maps) == config.n_layer
    assert maps[0].shape == (1, config.n_head, 4, 4)
    assert torch.all(maps[0][0, 0].triu(1) == 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": -1},
        {"max_new_tokens": 1, "temperature": 0},
        {"max_new_tokens": 1, "top_k": 0},
    ],
)
def test_generation_rejects_invalid_sampling_arguments(kwargs: dict[str, int | float]) -> None:
    config = ModelConfig(vocab_size=7, block_size=6, n_layer=1, n_head=1, n_embd=8)
    model = TinyTransformer(config)

    with pytest.raises(ValueError):
        model.generate(torch.zeros((1, 1), dtype=torch.long), **kwargs)
