from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("Cannot train a tokenizer on empty text")
        chars = sorted(set(text))
        stoi = {char: idx for idx, char in enumerate(chars)}
        itos = {idx: char for char, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[char] for char in text]
        except KeyError as exc:
            char = exc.args[0]
            raise ValueError(f"Character {char!r} is not in the tokenizer vocabulary") from exc

    def decode(self, ids: list[int]) -> str:
        try:
            return "".join(self.itos[idx] for idx in ids)
        except KeyError as exc:
            idx = exc.args[0]
            raise ValueError(f"Token id {idx!r} is not in the tokenizer vocabulary") from exc

    def to_dict(self) -> dict[str, dict[str, int] | dict[int, str]]:
        return {"stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_dict(cls, payload: dict[str, dict[str, int] | dict[int | str, str]]) -> "CharTokenizer":
        stoi = {str(char): int(idx) for char, idx in payload["stoi"].items()}
        itos = {int(idx): str(char) for idx, char in payload["itos"].items()}
        return cls(stoi=stoi, itos=itos)
