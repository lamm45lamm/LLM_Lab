"""
CHECKPOINT 1 — Vocab & Tokenizer (Phần 1.1-1.3, 2.1-2.3)

Chạy riêng:   pytest tests/test_part1.py -v
"""
from _target import target

SAMPLE_CORPUS = "This is a small sample text for testing the tokenizer implementation."


def test_1_1_build_vocab():
    tokens = ["the", "cat", "sat", "on", "the", "mat"]
    vocab = target.build_vocab(tokens)
    expected = {"cat": 0, "mat": 1, "on": 2, "sat": 3, "the": 4}
    assert vocab == expected


def test_1_2_simple_tokenizer_v1_encode():
    tokens = target.tokenize_text(SAMPLE_CORPUS)
    vocab = target.build_vocab(tokens)
    tokenizer = target.SimpleTokenizerV1(vocab)
    ids = tokenizer.encode(SAMPLE_CORPUS)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) == len(tokens)


def test_1_3_simple_tokenizer_v1_decode_roundtrip():
    tokens = target.tokenize_text(SAMPLE_CORPUS)
    vocab = target.build_vocab(tokens)
    tokenizer = target.SimpleTokenizerV1(vocab)
    ids = tokenizer.encode(SAMPLE_CORPUS)
    decoded = tokenizer.decode(ids)
    assert decoded == SAMPLE_CORPUS


def test_2_1_build_vocab_with_special_tokens():
    tokens = target.tokenize_text(SAMPLE_CORPUS)
    vocab = target.build_vocab_with_special_tokens(tokens)
    assert "<|endoftext|>" in vocab
    assert "<|unk|>" in vocab
    assert vocab["<|unk|>"] == len(vocab) - 1
    assert set(vocab.keys()) - {"<|endoftext|>", "<|unk|>"} == set(tokens)


def test_2_2_simple_tokenizer_v2_unknown_word():
    tokens = target.tokenize_text(SAMPLE_CORPUS)
    vocab = target.build_vocab_with_special_tokens(tokens)
    tokenizer = target.SimpleTokenizerV2(vocab)
    text_with_unknown = SAMPLE_CORPUS + " banana"  # "banana" không có trong vocab
    ids = tokenizer.encode(text_with_unknown)
    assert vocab["<|unk|>"] in ids


def test_2_3_simple_tokenizer_v2_decode():
    tokens = target.tokenize_text(SAMPLE_CORPUS)
    vocab = target.build_vocab_with_special_tokens(tokens)
    tokenizer = target.SimpleTokenizerV2(vocab)
    text_with_unknown = SAMPLE_CORPUS + " banana"
    ids = tokenizer.encode(text_with_unknown)
    decoded = tokenizer.decode(ids)
    assert "<|unk|>" in decoded
    assert "banana" not in decoded
