"""Tiny Transformer: a minimal GPT-style language model."""

from tiny_transformer.config import ModelConfig, TrainConfig
from tiny_transformer.tokenizer import BytePairTokenizer, CharTokenizer

__all__ = ["BytePairTokenizer", "CharTokenizer", "ModelConfig", "TinyTransformer", "TrainConfig"]


def __getattr__(name: str):
    if name == "TinyTransformer":
        from tiny_transformer.model import TinyTransformer

        return TinyTransformer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
