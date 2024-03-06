#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import numpy as np
import torch
import pytest

from probspecs import TrinaryLogic as TL


@pytest.mark.parametrize(
    "value,expected_result",
    [(0, TL.UNKNOWN), (1, TL.TRUE), (-1, TL.FALSE), (1.5, TL.TRUE), (-200.0, TL.FALSE)],
)
def test_creation(value, expected_result):
    assert TL(value) == expected_result


@pytest.mark.parametrize(
    "value,expected_result", [(TL.TRUE, True), (TL.UNKNOWN, False), (TL.FALSE, False)]
)
def test_is_true(value, expected_result):
    assert TL.is_true(value) == expected_result


@pytest.mark.parametrize("tensor,all_", [(torch.tensor, torch.all), (np.array, np.all)])
def test_is_true_batched(tensor, all_):
    value = tensor([TL.TRUE, TL.UNKNOWN, TL.FALSE])
    expected_result = tensor([True, False, False])
    assert all_(TL.is_true(value) == expected_result)


@pytest.mark.parametrize(
    "value,expected_result", [(TL.TRUE, False), (TL.UNKNOWN, False), (TL.FALSE, True)]
)
def test_is_false(value, expected_result):
    assert TL.is_false(value) == expected_result


@pytest.mark.parametrize("tensor,all_", [(torch.tensor, torch.all), (np.array, np.all)])
def test_is_false_batched(tensor, all_):
    value = tensor([TL.TRUE, TL.UNKNOWN, TL.FALSE])
    expected_result = tensor([False, False, True])
    assert all_(TL.is_false(value) == expected_result)


@pytest.mark.parametrize(
    "value,expected_result", [(TL.TRUE, False), (TL.UNKNOWN, True), (TL.FALSE, False)]
)
def test_is_unknown(value, expected_result):
    assert TL.is_unknown(value) == expected_result


@pytest.mark.parametrize("tensor,all_", [(torch.tensor, torch.all), (np.array, np.all)])
def test_is_unknown_batched(tensor, all_):
    value = tensor([TL.TRUE, TL.UNKNOWN, TL.FALSE])
    expected_result = tensor([False, True, False])
    assert all_(TL.is_unknown(value) == expected_result)


@pytest.mark.parametrize(
    "value,expected_result",
    [(TL.TRUE, TL.FALSE), (TL.UNKNOWN, TL.UNKNOWN), (TL.FALSE, TL.TRUE)],
)
def test_not_(value, expected_result):
    assert TL.not_(value) == expected_result


@pytest.mark.parametrize("tensor,all_", [(torch.tensor, torch.all), (np.array, np.all)])
def test_not_batched(tensor, all_):
    value = tensor([TL.TRUE, TL.UNKNOWN, TL.FALSE])
    expected_result = tensor([TL.FALSE, TL.UNKNOWN, TL.TRUE])
    assert all_(TL.not_(value) == expected_result)


@pytest.mark.parametrize(
    "values,expected_result",
    [
        ((TL.TRUE, TL.TRUE), TL.TRUE),
        ((TL.TRUE, TL.FALSE), TL.FALSE),
        ((TL.TRUE, TL.UNKNOWN), TL.UNKNOWN),
        ((TL.FALSE, TL.TRUE), TL.FALSE),
        ((TL.FALSE, TL.FALSE), TL.FALSE),
        ((TL.FALSE, TL.UNKNOWN), TL.FALSE),
        ((TL.UNKNOWN, TL.TRUE), TL.UNKNOWN),
        ((TL.UNKNOWN, TL.FALSE), TL.FALSE),
        ((TL.UNKNOWN, TL.UNKNOWN), TL.UNKNOWN),
        ((TL.TRUE, TL.TRUE, TL.TRUE), TL.TRUE),
        ((TL.TRUE, TL.FALSE, TL.TRUE), TL.FALSE),
        ((TL.TRUE, TL.UNKNOWN, TL.TRUE), TL.UNKNOWN),
        ((TL.UNKNOWN, TL.TRUE, TL.TRUE), TL.UNKNOWN),
        ((TL.FALSE, TL.TRUE, TL.TRUE), TL.FALSE),
        ((TL.TRUE, TL.TRUE, TL.UNKNOWN), TL.UNKNOWN),
        ((TL.FALSE, TL.UNKNOWN, TL.UNKNOWN), TL.FALSE),
        ((TL.TRUE,) * 100, TL.TRUE),
        ((TL.TRUE,) * 15 + (TL.FALSE,) * 2 + (TL.TRUE,) * 9, TL.FALSE),
        ((TL.UNKNOWN,) + (TL.TRUE,) * 72 + (TL.FALSE,) * 2, TL.FALSE),
        ((), TL.TRUE),
    ],
)
def test_and_(values, expected_result):
    assert TL.and_(*values) == expected_result


@pytest.mark.parametrize("tensor,all_", [(torch.tensor, torch.all), (np.array, np.all)])
def test_and_batched_1(tensor, all_):
    values = [
        tensor([TL.TRUE] * 3 + [TL.FALSE] * 3 + [TL.UNKNOWN] * 3),
        tensor([TL.TRUE, TL.FALSE, TL.UNKNOWN] * 3),
    ]
    expected_result = tensor(
        [TL.TRUE, TL.FALSE, TL.UNKNOWN]
        + [TL.FALSE] * 3
        + [TL.UNKNOWN]
        + [TL.FALSE]
        + [TL.UNKNOWN]
    )
    assert all_(TL.and_(*values) == expected_result)


@pytest.mark.parametrize("tensor,all_", [(torch.tensor, torch.all), (np.array, np.all)])
def test_and_batched_2(tensor, all_):
    values = [
        tensor([TL.TRUE, TL.TRUE, TL.TRUE, TL.FALSE, TL.FALSE, TL.UNKNOWN]),
        tensor([TL.TRUE, TL.TRUE, TL.FALSE, TL.FALSE, TL.TRUE, TL.TRUE]),
        tensor([TL.TRUE, TL.FALSE, TL.UNKNOWN, TL.TRUE, TL.UNKNOWN, TL.TRUE]),
    ]
    expected_result = tensor(
        [TL.TRUE, TL.FALSE, TL.FALSE, TL.FALSE, TL.FALSE, TL.UNKNOWN]
    )
    assert all_(TL.and_(*values) == expected_result)


@pytest.mark.parametrize(
    "values,expected_result",
    [
        ((TL.TRUE, TL.TRUE), TL.TRUE),
        ((TL.TRUE, TL.FALSE), TL.TRUE),
        ((TL.TRUE, TL.UNKNOWN), TL.TRUE),
        ((TL.FALSE, TL.TRUE), TL.TRUE),
        ((TL.FALSE, TL.FALSE), TL.FALSE),
        ((TL.FALSE, TL.UNKNOWN), TL.UNKNOWN),
        ((TL.UNKNOWN, TL.TRUE), TL.TRUE),
        ((TL.UNKNOWN, TL.FALSE), TL.UNKNOWN),
        ((TL.UNKNOWN, TL.UNKNOWN), TL.UNKNOWN),
        ((TL.TRUE, TL.TRUE, TL.TRUE), TL.TRUE),
        ((TL.TRUE, TL.FALSE, TL.TRUE), TL.TRUE),
        ((TL.TRUE, TL.UNKNOWN, TL.TRUE), TL.TRUE),
        ((TL.UNKNOWN, TL.TRUE, TL.TRUE), TL.TRUE),
        ((TL.FALSE, TL.FALSE, TL.FALSE), TL.FALSE),
        ((TL.TRUE, TL.TRUE, TL.UNKNOWN), TL.TRUE),
        ((TL.FALSE, TL.UNKNOWN, TL.UNKNOWN), TL.UNKNOWN),
        ((TL.FALSE,) * 100, TL.FALSE),
        ((TL.TRUE,) * 15 + (TL.FALSE,) * 2 + (TL.TRUE,) * 9, TL.TRUE),
        ((TL.UNKNOWN,) + (TL.TRUE,) * 72 + (TL.FALSE,) * 2, TL.TRUE),
        ((), TL.FALSE),
    ],
)
def test_or_(values, expected_result):
    assert TL.or_(*values) == expected_result


@pytest.mark.parametrize("tensor,all_", [(torch.tensor, torch.all), (np.array, np.all)])
def test_or_batched_1(tensor, all_):
    values = [
        tensor([TL.TRUE] * 3 + [TL.FALSE] * 3 + [TL.UNKNOWN] * 3),
        tensor([TL.TRUE, TL.FALSE, TL.UNKNOWN] * 3),
    ]
    expected_result = tensor(
        [TL.TRUE] * 4 + [TL.FALSE, TL.UNKNOWN, TL.TRUE] + [TL.UNKNOWN] * 2
    )
    assert all_(TL.or_(*values) == expected_result)


@pytest.mark.parametrize("tensor,all_", [(torch.tensor, torch.all), (np.array, np.all)])
def test_not_batched_2(tensor, all_):
    values = [
        tensor([TL.FALSE, TL.FALSE, TL.TRUE, TL.FALSE, TL.FALSE, TL.UNKNOWN]),
        tensor([TL.FALSE, TL.TRUE, TL.FALSE, TL.FALSE, TL.TRUE, TL.TRUE]),
        tensor([TL.FALSE, TL.FALSE, TL.UNKNOWN, TL.TRUE, TL.UNKNOWN, TL.TRUE]),
    ]
    expected_result = tensor([TL.FALSE, TL.TRUE, TL.TRUE, TL.TRUE, TL.TRUE, TL.TRUE])
    assert all_(TL.or_(*values) == expected_result)


if __name__ == "__main__":
    pytest.main()
