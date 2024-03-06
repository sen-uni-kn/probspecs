# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import pytest
import torch

from probspecs import TensorInputSpace, TabularInputSpace


def test_tensor_space():
    torch.manual_seed(991187180197848)
    mid = 100 * torch.rand((3, 32, 32)) - 50  # mid point
    ran = 100 * torch.rand((3, 32, 32))  # range
    lb = mid - ran
    ub = mid + ran

    in_space = TensorInputSpace(lb, ub)
    assert in_space.input_shape == (3, 32, 32)
    assert torch.allclose(in_space.input_bounds[0], lb)
    assert torch.allclose(in_space.input_bounds[1], ub)


@pytest.fixture
def mtcars():
    torch.manual_seed(685723051036806)
    cont = TabularInputSpace.AttributeType.CONTINUOUS
    cat = TabularInputSpace.AttributeType.CATEGORICAL
    int_ = TabularInputSpace.AttributeType.INTEGER
    return TabularInputSpace(
        attributes=(
            "mpg",
            "cyl",
            "disp",
            "hp",
            "vs",
        ),
        data_types={"mpg": cont, "cyl": cat, "disp": cont, "hp": int_, "vs": cat},
        continuous_ranges={"mpg": (10.0, 34.0), "disp": (50.0, 500.0)},
        integer_ranges={"hp": (52, 335)},
        categorical_values={"cyl": ("4", "6", "8"), "vs": ("V", "straight")},
    )


def test_tabular_space_base(mtcars):
    in_space = mtcars
    cont = TabularInputSpace.AttributeType.CONTINUOUS
    cat = TabularInputSpace.AttributeType.CATEGORICAL
    int_ = TabularInputSpace.AttributeType.INTEGER

    assert in_space.attribute_names == ("mpg", "cyl", "disp", "hp", "vs")
    assert in_space.attribute_types == (cont, cat, cont, int_, cat)

    assert in_space.attribute_name(0) == "mpg"
    assert in_space.attribute_name(4) == "vs"
    assert in_space.attribute_type(1) == cat
    assert in_space.attribute_type(2) == cont

    assert in_space.attribute_type("mpg") == cont
    assert in_space.attribute_type("vs") == cat

    assert in_space.input_shape == (8,)


def test_tabular_space_bounds(mtcars):
    in_space = mtcars
    assert in_space.attribute_bounds(2) == (50.0, 500.0)
    with pytest.raises(ValueError):
        in_space.attribute_bounds(1)

    assert torch.allclose(
        in_space.input_bounds[0],
        torch.tensor([10.0, 0.0, 0.0, 0.0, 50.0, 52.0, 0.0, 0.0]),
    )
    assert torch.allclose(
        in_space.input_bounds[1],
        torch.tensor([34.0, 1.0, 1.0, 1.0, 500.0, 335.0, 1.0, 1.0]),
    )


def test_tabular_space_values(mtcars):
    in_space = mtcars
    assert in_space.attribute_values(4) == ("V", "straight")
    assert in_space.attribute_values("vs") == ("V", "straight")
    assert in_space.attribute_values(3) == tuple(range(52, 335 + 1))
    assert in_space.attribute_values("hp") == tuple(range(52, 335 + 1))
    with pytest.raises(ValueError):
        in_space.attribute_values(0)


def test_tabular_space_encoding_layout(mtcars):
    in_space = mtcars
    assert in_space.encoding_layout == {
        "mpg": 0,
        "cyl": {"4": 1, "6": 2, "8": 3},
        "disp": 4,
        "hp": 5,
        "vs": {"V": 6, "straight": 7},
    }


def test_tabular_space_encode_decode(mtcars):
    in_space = mtcars
    maserati = {"mpg": 15.0, "cyl": "8", "disp": 301.0, "hp": 335, "vs": "V"}

    encoding = in_space.encode(list(maserati.values()))
    assert torch.allclose(
        encoding, torch.tensor([15.0, 0.0, 0.0, 1.0, 301.0, 335.0, 1.0, 0.0])
    )
    assert in_space.decode(encoding) == tuple(maserati.values())


if __name__ == "__main__":
    pytest.main()
