from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol


class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def id_to_token(self, idx: int) -> str: ...

    def to_dict(self) -> dict[str, object]: ...


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

    def id_to_token(self, idx: int) -> str:
        try:
            return self.itos[idx]
        except KeyError as exc:
            raise ValueError(f"Token id {idx!r} is not in the tokenizer vocabulary") from exc

    def to_dict(self) -> dict[str, object]:
        return {"type": "char", "stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_dict(cls, payload: dict[str, dict[str, int] | dict[int | str, str]]) -> "CharTokenizer":
        stoi = {str(char): int(idx) for char, idx in payload["stoi"].items()}
        itos = {int(idx): str(char) for idx, char in payload["itos"].items()}
        return cls(stoi=stoi, itos=itos)


@dataclass(frozen=True)
class BytePairTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]
    merges: list[tuple[str, str]]

    @classmethod
    def train(cls, text: str, vocab_size: int = 256) -> "BytePairTokenizer":
        if not text:
            raise ValueError("Cannot train a tokenizer on empty text")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        base_vocab = sorted(set(text))
        sequences = [[char for char in text]]
        merges: list[tuple[str, str]] = []
        vocab = set(base_vocab)

        while len(vocab) < vocab_size:
            pair_counts = _count_pairs(sequences)
            if not pair_counts:
                break
            pair, count = pair_counts.most_common(1)[0]
            merged = "".join(pair)
            if count < 2 or merged in vocab:
                break
            sequences = [_merge_pair(sequence, pair, merged) for sequence in sequences]
            merges.append(pair)
            vocab.add(merged)

        tokens = sorted(vocab, key=lambda token: (len(token), token))
        stoi = {token: idx for idx, token in enumerate(tokens)}
        itos = {idx: token for token, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos, merges=merges)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - {token for token in self.stoi if len(token) == 1})
        if unknown:
            raise ValueError(f"Characters {unknown!r} are not in the tokenizer vocabulary")

        pieces = list(text)
        for left, right in self.merges:
            pieces = _merge_pair(pieces, (left, right), left + right)
        return [self.stoi[piece] for piece in pieces]

    def decode(self, ids: list[int]) -> str:
        try:
            return "".join(self.itos[idx] for idx in ids)
        except KeyError as exc:
            idx = exc.args[0]
            raise ValueError(f"Token id {idx!r} is not in the tokenizer vocabulary") from exc

    def id_to_token(self, idx: int) -> str:
        try:
            return self.itos[idx]
        except KeyError as exc:
            raise ValueError(f"Token id {idx!r} is not in the tokenizer vocabulary") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "bpe",
            "stoi": self.stoi,
            "itos": self.itos,
            "merges": self.merges,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BytePairTokenizer":
        raw_stoi = payload["stoi"]
        raw_itos = payload["itos"]
        raw_merges = payload["merges"]
        if not isinstance(raw_stoi, dict) or not isinstance(raw_itos, dict):
            raise ValueError("Invalid BPE tokenizer payload")
        if not isinstance(raw_merges, list):
            raise ValueError("Invalid BPE merge payload")
        stoi = {str(token): int(idx) for token, idx in raw_stoi.items()}
        itos = {int(idx): str(token) for idx, token in raw_itos.items()}
        merges = [(str(left), str(right)) for left, right in raw_merges]
        return cls(stoi=stoi, itos=itos, merges=merges)


def tokenizer_from_dict(payload: dict[str, object]) -> Tokenizer:
    tokenizer_type = payload.get("type", "char")
    if tokenizer_type == "char":
        return CharTokenizer.from_dict(payload)  # type: ignore[arg-type]
    if tokenizer_type == "bpe":
        return BytePairTokenizer.from_dict(payload)
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type!r}")


def _count_pairs(sequences: list[list[str]]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for sequence in sequences:
        counts.update(zip(sequence, sequence[1:]))
    return counts


def _merge_pair(sequence: list[str], pair: tuple[str, str], merged: str) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(sequence):
        if idx < len(sequence) - 1 and (sequence[idx], sequence[idx + 1]) == pair:
            out.append(merged)
            idx += 2
        else:
            out.append(sequence[idx])
            idx += 1
    return out
