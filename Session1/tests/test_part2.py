"""
CHECKPOINT 2 — BPE (tiktoken) & trực giác sliding window (Phần 3.1, 4.1)

Chạy riêng:   pytest tests/test_part2.py -v

Lưu ý: test_3_1 cần internet ở LẦN CHẠY ĐẦU TIÊN để tiktoken tải file vocab
GPT-2 (sau đó sẽ được cache lại trên máy).
"""
from _target import target


def test_3_1_bpe_roundtrip():
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces."
    ids, decoded = target.bpe_roundtrip_demo(text)
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)
    assert decoded == text


def test_3_1_bpe_handles_unknown_words_without_unk_token():
    # BPE tách được cả từ "lạ" (không có trong tiếng Anh thông thường) thành
    # các đơn vị con, không cần token <|unk|>.
    text = "someunknownPlace is a made-up word."
    ids, decoded = target.bpe_roundtrip_demo(text)
    assert len(ids) >= len(text.split())  # bị tách thành nhiều subword
    assert decoded == text


def test_4_1_next_token_pairs_basic():
    pairs = target.get_next_token_pairs([10, 20, 30, 40, 50], context_size=4)
    expected = [
        ([10], 20),
        ([10, 20], 30),
        ([10, 20, 30], 40),
        ([10, 20, 30, 40], 50),
    ]
    assert pairs == expected


def test_4_1_next_token_pairs_length():
    token_ids = list(range(100, 110))
    context_size = 5
    pairs = target.get_next_token_pairs(token_ids, context_size)
    assert len(pairs) == context_size
    for i, (context, tgt) in enumerate(pairs, start=1):
        assert context == token_ids[:i]
        assert tgt == token_ids[i]
