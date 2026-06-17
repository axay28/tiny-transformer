import pytest

from tiny_transformer.tokenizer import CharTokenizer


def test_tokenizer_round_trip() -> None:
    tokenizer = CharTokenizer.train("hello transformer")
    ids = tokenizer.encode("hello")

    assert tokenizer.decode(ids) == "hello"
    assert tokenizer.vocab_size == len(set("hello transformer"))


def test_tokenizer_rejects_unknown_characters() -> None:
    tokenizer = CharTokenizer.train("abc")

    with pytest.raises(ValueError, match="not in the tokenizer vocabulary"):
        tokenizer.encode("abcd")
