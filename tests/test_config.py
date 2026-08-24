import pytest

from tiny_transformer.config import ModelConfig, TrainConfig


@pytest.mark.parametrize(
    "overrides",
    [
        {"vocab_size": 0},
        {"block_size": 0},
        {"n_layer": 0},
        {"n_head": 0},
        {"n_embd": 0},
        {"dropout": -0.1},
        {"dropout": 1.0},
    ],
)
def test_model_config_rejects_invalid_values(overrides: dict[str, int | float]) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**{"vocab_size": 10, **overrides})


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size": 0},
        {"learning_rate": 0},
        {"max_steps": 0},
        {"eval_interval": 0},
        {"eval_batches": 0},
        {"grad_accum_steps": 0},
    ],
)
def test_train_config_rejects_invalid_values(overrides: dict[str, int | float]) -> None:
    with pytest.raises(ValueError):
        TrainConfig(**overrides)
