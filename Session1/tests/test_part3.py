"""
CHECKPOINT 3 — GPTDatasetV1 & create_dataloader_v1 (Phần 5.1-5.3, 6.1)

Chạy riêng:   pytest tests/test_part3.py -v

Các test dưới đây dùng một tokenizer GIẢ LẬP (DummyTokenizer) thay vì
tiktoken thật, để: (1) kết quả hoàn toàn xác định (deterministic), và
(2) không cần internet khi chạy checkpoint này.
"""
import torch

from _target import target


class DummyTokenizer:
    """Tokenizer giả lập: luôn trả về đúng 20 token id cố định [1..20],
    bất kể nội dung `text` truyền vào là gì. Dùng để kiểm tra logic cắt
    sliding-window của GPTDatasetV1 một cách xác định."""

    def encode(self, text, allowed_special=None):
        return list(range(1, 21))


def test_5_1_dataset_length_and_windowing():
    dataset = target.GPTDatasetV1("nội dung không quan trọng", DummyTokenizer(),
                                   max_length=4, stride=4)
    # 20 token, max_length=4, stride=4 -> range(0, 16, 4) -> 4 cửa sổ
    assert len(dataset) == 4


def test_5_2_len_matches_input_ids():
    dataset = target.GPTDatasetV1("x", DummyTokenizer(), max_length=4, stride=4)
    assert len(dataset) == len(dataset.input_ids) == len(dataset.target_ids)


def test_5_3_getitem_values_and_shift():
    dataset = target.GPTDatasetV1("x", DummyTokenizer(), max_length=4, stride=4)
    x0, y0 = dataset[0]
    assert torch.equal(x0, torch.tensor([1, 2, 3, 4]))
    assert torch.equal(y0, torch.tensor([2, 3, 4, 5]))

    x1, y1 = dataset[1]
    assert torch.equal(x1, torch.tensor([5, 6, 7, 8]))
    assert torch.equal(y1, torch.tensor([6, 7, 8, 9]))


def test_5_1_overlapping_windows_with_smaller_stride():
    dataset = target.GPTDatasetV1("x", DummyTokenizer(), max_length=4, stride=1)
    # range(0, 16, 1) -> 16 cửa sổ, chồng lấn lên nhau
    assert len(dataset) == 16
    x0, _ = dataset[0]
    x1, _ = dataset[1]
    assert torch.equal(x0[1:], x1[:-1])  # 2 cửa sổ liên tiếp lệch nhau 1 vị trí


def test_6_1_create_dataloader_v1(monkeypatch):
    monkeypatch.setattr(target.tiktoken, "get_encoding", lambda name: DummyTokenizer())

    dataloader = target.create_dataloader_v1(
        "nội dung không quan trọng", batch_size=2, max_length=4, stride=4,
        shuffle=False, drop_last=True,
    )
    inputs, targets = next(iter(dataloader))

    assert inputs.shape == (2, 4)
    assert targets.shape == (2, 4)
    # bất biến: target = input dịch phải 1 vị trí, trong cùng 1 cửa sổ
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
