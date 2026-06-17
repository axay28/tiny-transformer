import pytest

from tiny_transformer.tokenizer import BytePairTokenizer, CharTokenizer, tokenizer_from_dict


def test_tokenizer_round_trip() -> None:
    tokenizer = CharTokenizer.train("hello transformer")
    ids = tokenizer.encode("hello")

    assert tokenizer.decode(ids) == "hello"
    assert tokenizer.vocab_size == len(set("hello transformer"))


def test_tokenizer_rejects_unknown_characters() -> None:
    tokenizer = CharTokenizer.train("abc")

    with pytest.raises(ValueError, match="not in the tokenizer vocabulary"):
        tokenizer.encode("abcd")


def test_bpe_tokenizer_round_trip_and_serialization() -> None:
    tokenizer = BytePairTokenizer.train("banana bandana", vocab_size=20)
    ids = tokenizer.encode("banana")
    restored = tokenizer_from_dict(tokenizer.to_dict())

    assert tokenizer.decode(ids) == "banana"
    assert restored.decode(restored.encode("bandana")) == "bandana"
    assert tokenizer.vocab_size > len(set("banana bandana"))
