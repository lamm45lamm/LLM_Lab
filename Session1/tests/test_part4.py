"""
CHECKPOINT 4 — Embedding & Pipeline hoàn chỉnh (Phần 7.1, 8.1)

Chạy riêng:   pytest tests/test_part4.py -v
"""
import torch

from _target import target


class DummyTokenizer:
    """Tokenizer giả lập, trả về 100 token id cố định — đủ để tạo nhiều
    batch mà không cần internet / tiktoken thật."""

    def encode(self, text, allowed_special=None):
        return list(range(1, 101))


def test_7_1_token_embedding_shapes():
    input_ids = torch.tensor([2, 3, 5, 1])
    weight, embedded = target.demo_token_embedding(
        vocab_size=6, output_dim=3, input_ids=input_ids
    )
    assert tuple(weight.shape) == (6, 3)
    assert tuple(embedded.shape) == (4, 3)


def test_7_1_token_embedding_is_a_lookup():
    # Bất biến quan trọng nhất của Embedding: embedded[i] phải bằng đúng
    # hàng thứ input_ids[i] của ma trận trọng số.
    input_ids = torch.tensor([2, 3, 5, 1])
    weight, embedded = target.demo_token_embedding(
        vocab_size=6, output_dim=3, input_ids=input_ids
    )
    for i, token_id in enumerate(input_ids.tolist()):
        assert torch.equal(embedded[i], weight[token_id])


def test_8_1_build_input_embeddings_shape(monkeypatch):
    monkeypatch.setattr(target.tiktoken, "get_encoding", lambda name: DummyTokenizer())

    result = target.build_input_embeddings(
        "nội dung không quan trọng", vocab_size=50257, output_dim=256,
        max_length=4, batch_size=8,
    )
    assert tuple(result.shape) == (8, 4, 256)


def test_8_1_build_input_embeddings_is_torch_tensor(monkeypatch):
    monkeypatch.setattr(target.tiktoken, "get_encoding", lambda name: DummyTokenizer())

    result = target.build_input_embeddings(
        "nội dung không quan trọng", vocab_size=50257, output_dim=256,
        max_length=4, batch_size=8,
    )
    assert isinstance(result, torch.Tensor)
