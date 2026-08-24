from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 1_000
    eval_interval: int = 100
    eval_batches: int = 20
    grad_accum_steps: int = 1
    use_amp: bool = False
    seed: int = 1337
    output_path: str = "runs/tiny-transformer.pt"
    loss_history_path: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.eval_batches <= 0:
            raise ValueError("eval_batches must be positive")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")

    def to_dict(self) -> dict[str, bool | int | float | str | None]:
        return asdict(self)
