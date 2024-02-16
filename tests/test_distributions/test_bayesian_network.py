#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import numpy as np
from scipy.stats import truncnorm, bernoulli
import pytest
import torch

from probspecs import (
    BayesianNetwork,
    CategoricalOneHot,
    UnivariateContinuousDistribution,
    UnivariateDiscreteDistribution,
    MultivariateIndependent,
)


def test_create_bayes_net_1():
    factory = BayesianNetwork.Factory()
    source = factory.new_node("X")
    distribution = truncnorm(a=-3.0, b=3.0, loc=0.0, scale=1.0)
    distribution = UnivariateContinuousDistribution(distribution)
    source.continuous_event_space(lower=-3.0, upper=3.0)
    source.set_conditional_probability({}, distribution)

    sink = factory.new_node("Y")
    sink.add_parent(source)
    sink.continuous_event_space(lower=-16.0, upper=16.0)
    sink.set_conditional_probability(
        {source: (-3.0, 0.0)},
        distribution=UnivariateContinuousDistribution(
            truncnorm(a=-1.0, b=1.0, loc=-15.0, scale=1.0)
        ),
    )
    sink.set_conditional_probability(
        {source: (0.0, 3.0)},
        distribution=UnivariateContinuousDistribution(
            truncnorm(a=-1.0, b=1.0, loc=15.0, scale=1.0)
        ),
    )

    bayes_net = factory.create()
    print(bayes_net)
    return bayes_net


def test_create_bayes_net_2():
    factory = BayesianNetwork.Factory()
    n1 = factory.new_node("n1")
    n2 = factory.new_node("n2")

    n1.discrete_event_space([1.0, 0.0], [0.0, 1.0])
    n1.set_conditional_probability({}, CategoricalOneHot(torch.tensor([0.3, 0.7])))

    n2.add_parent(n1)
    n2.discrete_event_space([1.0, 0.0], [0.0, 1.0])
    n2.set_conditional_probability(
        {"n1": torch.tensor([1.0, 0.0])},
        CategoricalOneHot([0.1, 0.9]),
    )
    n2.set_conditional_probability(
        {"n1": [0.0, 1.0]},
        CategoricalOneHot([0.5, 0.5]),
    )

    bayes_net = factory.create()
    print(bayes_net)
    return bayes_net


def test_create_bayes_net_3():
    factory = BayesianNetwork.Factory()
    n1 = factory.new_node("n1")
    n2 = factory.new_node("n2")
    n3 = factory.new_node("n3")

    n1.continuous_event_space(-3.0, 3.0)
    n1.set_conditional_probability(
        {}, UnivariateContinuousDistribution(truncnorm(a=-3.0, b=3.0))
    )

    n2.add_parent(n1)
    n2.discrete_event_space([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    n2.set_conditional_probability(
        {"n1": (torch.tensor([-3.0]), torch.tensor([0.0]))},
        CategoricalOneHot(torch.tensor([0.0, 0.25, 0.75])),
    )
    n2.set_conditional_probability(
        {"n1": (torch.tensor([0.0]), torch.tensor([3.0]))},
        CategoricalOneHot(torch.tensor([0.9, 0.1, 0.0])),
    )

    n3.add_parent(n2)
    n3.continuous_event_space(-1.0, 1001.0)
    n3.set_conditional_probability(
        {"n2": torch.tensor([1.0, 0.0, 0.0])},
        UnivariateContinuousDistribution(truncnorm(a=-1, b=1, loc=0)),
    )
    n3.set_conditional_probability(
        {"n2": torch.tensor([0.0, 1.0, 0.0])},
        UnivariateContinuousDistribution(truncnorm(a=-1, b=1, loc=100)),
    )
    n3.set_conditional_probability(
        {"n2": torch.tensor([0.0, 0.0, 1.0])},
        UnivariateContinuousDistribution(truncnorm(a=-1, b=1, loc=1000)),
    )

    bayes_net = factory.create()
    print(bayes_net)
    return bayes_net


def test_create_bayes_net_4():
    factory = BayesianNetwork.Factory()
    x, y, z = factory.new_nodes("x", "y", "z")

    x.discrete_event_space(0.0, 1.0)
    y.discrete_event_space(0.0, 1.0)
    x.set_conditional_probability({}, UnivariateDiscreteDistribution(bernoulli(p=0.5)))
    y.set_conditional_probability({}, UnivariateDiscreteDistribution(bernoulli(p=0.1)))

    def make_2d_truncnorm(loc1, loc2, loc3, loc4):
        norms = [
            truncnorm(a=-3.0, b=3.0, loc=loc, scale=1.0)
            for loc in (loc1, loc2, loc3, loc4)
        ]
        norms = [UnivariateDiscreteDistribution(d) for d in norms]
        return MultivariateIndependent(*norms, event_shape=(2, 2))

    z.set_parents(x, y)
    z.continuous_event_space(
        [[-103.0, -103.0], [-103.0, -103.0]], [[13.0, 13.0], [13.0, 13.0]]
    )
    z.set_conditional_probability(
        {
            x: (torch.tensor([0.0]), torch.tensor([0.0])),
            y: (torch.tensor([0.0]), torch.tensor([0.0])),
        },
        make_2d_truncnorm(0.0, 0.0, 0.0, 0.0),
    )
    z.set_conditional_probability(
        {
            x: (torch.tensor([0.0]), torch.tensor([0.0])),
            y: (torch.tensor([1.0]), torch.tensor([1.0])),
        },
        make_2d_truncnorm(10.0, 10.0, 10.0, 10.0),
    )
    z.set_conditional_probability(
        {
            x: (torch.tensor([1.0]), torch.tensor([1.0])),
            y: (torch.tensor([0.0]), torch.tensor([0.0])),
        },
        make_2d_truncnorm(-10.0, -10.0, -10.0, -10.0),
    )
    z.set_conditional_probability(
        {
            x: (torch.tensor([1.0]), torch.tensor([1.0])),
            y: (torch.tensor([1.0]), torch.tensor([1.0])),
        },
        make_2d_truncnorm(-100.0, -100.0, -100.0, -100.0),
    )

    bayes_net = factory.create()
    print(bayes_net)
    return bayes_net


def test_create_no_duplicate_names():
    factory = BayesianNetwork.Factory()
    n1 = factory.new_node("n1")
    n2 = factory.new_node("n2")
    with pytest.raises(ValueError):
        n3 = factory.new_node("n1")


@pytest.mark.parametrize(
    "other_event",
    [
        ([0.0, 0.0], [1.0, 1.0]),
        ([0.1, 0.1], [0.5, 0.2]),
        ([-1.0, 0.5], [0.74, 0.43]),
        ([0.3, 0.215], [2.0, 0.5]),
        ([-0.1, 0.5], [1.5, 0.6]),
        ([0.25, -1.0], [0.75, 0.33]),
    ],
)
def test_create_not_disjoint(other_event):
    other_event = tuple(torch.tensor(elem) for elem in other_event)

    factory = BayesianNetwork.Factory()
    n1 = factory.new_node("n1")
    n2 = factory.new_node("n2")

    n2.add_parent(n1)
    n2.set_conditional_probability(
        {"n1": (torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))},
        UnivariateContinuousDistribution(truncnorm(a=0.0, b=1.0)),
    )
    with pytest.raises(ValueError):
        n2.set_conditional_probability(
            {"n1": other_event},
            UnivariateContinuousDistribution(truncnorm(a=-1.0, b=0.0)),
        )


@pytest.mark.parametrize(
    "other_event",
    [
        ([-1.0, -1.0], [-0.5, -0.5]),
        ([-1.0, -1.0], [0.0, 0.0]),
        ([0.25, -1.0], [0.75, -0.33]),
        ([1.1, 0.3], [1.5, 0.4]),
        ([1.0, 0.0], [1.0, 1.0]),
    ],
)
def test_create_actually_disjoint(other_event):
    other_event = tuple(torch.tensor(elem) for elem in other_event)

    factory = BayesianNetwork.Factory()
    n1 = factory.new_node("n1")
    n2 = factory.new_node("n2")

    n2.add_parent(n1)
    n2.set_conditional_probability(
        {"n1": (torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))},
        UnivariateContinuousDistribution(truncnorm(a=0.0, b=1.0)),
    )
    n2.set_conditional_probability(
        {"n1": other_event},
        UnivariateContinuousDistribution(truncnorm(a=-1.0, b=0.0)),
    )


def test_create_identical_condition():
    factory = BayesianNetwork.Factory()
    n1 = factory.new_node("n1")
    n2 = factory.new_node("n2")

    n1.set_conditional_probability(
        {}, CategoricalOneHot(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    )

    n2.add_parent(n1)
    n2.set_conditional_probability(
        {
            "n1": (
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
            )
        },
        UnivariateContinuousDistribution(truncnorm(a=-1, b=1, loc=0)),
    )
    with pytest.raises(ValueError):
        n2.set_conditional_probability(
            {
                "n1": (
                    torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    torch.tensor([1.0, 0.0, 0.0, 0.0]),
                )
            },
            UnivariateContinuousDistribution(truncnorm(a=-1, b=1, loc=100)),
        )


def test_create_no_event_shape_mismatch():
    factory = BayesianNetwork.Factory()
    n1 = factory.new_node("n1")
    n2 = factory.new_node("n2")

    n2.add_parent(n1)
    n2.set_conditional_probability(
        {"n1": (torch.tensor([0.0]), torch.tensor([1.0]))},
        UnivariateContinuousDistribution(truncnorm(a=0.0, b=1.0)),
    )
    with pytest.raises(ValueError):
        n2.set_conditional_probability(
            {"n1": (torch.tensor([-1.0]), torch.tensor([-0.1]))},
            CategoricalOneHot(torch.tensor([0.1, 0.6, 0.2])),
        )


def test_create_not_entire_parents_space_covered_1():
    factory = BayesianNetwork.Factory()
    source = factory.new_node("X")
    distribution = truncnorm(a=-3.0, b=3.0, loc=0.0, scale=1.0)
    distribution = UnivariateContinuousDistribution(distribution)
    source.set_conditional_probability({}, distribution)

    sink = factory.new_node("Y")
    sink.add_parent(source)
    sink.set_conditional_probability(
        {source: (-3.0, 0.0)},
        distribution=UnivariateContinuousDistribution(
            truncnorm(a=-1.0, b=1.0, loc=-15.0, scale=1.0)
        ),
    )
    sink.set_conditional_probability(
        {source: (0.0, 3.0)},
        distribution=UnivariateContinuousDistribution(
            truncnorm(a=-1.0, b=1.0, loc=15.0, scale=1.0)
        ),
    )

    with pytest.raises(ValueError):
        factory.create()


def test_create_not_entire_parents_space_covered_2():
    factory = BayesianNetwork.Factory()
    n1 = factory.new_node("n1")
    n2 = factory.new_node("n2")

    n1.discrete_event_space([1.0, 0.0], [0.0, 1.0])
    n1.set_conditional_probability({}, CategoricalOneHot(torch.tensor([0.3, 0.7])))

    n2.add_parent(n1)
    n2.discrete_event_space([1.0, 0.0], [0.0, 1.0])
    n2.set_conditional_probability(
        {"n1": (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))},
        CategoricalOneHot(torch.tensor([0.1, 0.9])),
    )

    with pytest.raises(ValueError):
        factory.create()


def test_create_wrong_bounds_shape():
    factory = BayesianNetwork.Factory()
    n1 = factory.new_node("n1")
    n2 = factory.new_node("n2")

    n1.discrete_event_space([1.0, 0.0], [0.0, 1.0])
    n1.set_conditional_probability({}, CategoricalOneHot(torch.tensor([0.3, 0.7])))

    n2.add_parent(n1)
    n2.discrete_event_space([1.0, 0.0], [0.0, 1.0])
    n2.set_conditional_probability(
        {"n1": torch.tensor([1.0, 0.0])},
        CategoricalOneHot([0.1, 0.9]),
    )
    n2.set_conditional_probability(
        {"n1": (0.0, 1.0)},  # this is not a single tensor! These are bounds.
        CategoricalOneHot([0.5, 0.5]),
    )

    with pytest.raises(ValueError):
        factory.create()


bayes_net_1 = pytest.fixture(test_create_bayes_net_1)
bayes_net_2 = pytest.fixture(test_create_bayes_net_2)
bayes_net_3 = pytest.fixture(test_create_bayes_net_3)
bayes_net_4 = pytest.fixture(test_create_bayes_net_4)


def test_sample_1(bayes_net_1):
    torch.manual_seed(527942209811048)
    np.random.seed(8995286)

    x = bayes_net_1.sample(10)
    assert torch.all(x[x[:, 0] > 0.0, 1] >= 10.0)
    assert torch.all(x[x[:, 0] < 0.0, 1] <= 10.0)


def test_sample_2(bayes_net_3):
    torch.manual_seed(567942209811048)
    np.random.seed(118995286)

    x = bayes_net_3.sample(1000)
    assert torch.all(x[x[:, 0] < 0.0, 1] == 0.0)
    assert torch.all(x[x[:, 0] > 0.0, 3] == 0.0)

    assert torch.all(x[x[:, 1] == 1.0, 4] < 5.0)
    assert torch.all(x[x[:, 2] == 1.0, 4] > 5.0)
    assert torch.all(x[x[:, 2] == 1.0, 4] < 500.0)
    assert torch.all(x[x[:, 3] == 1.0, 4] > 500.0)

    assert torch.all(x[x[:, 0] < 0.0, 4] > 5.0)
    assert torch.all(x[x[:, 0] > 0.0, 4] < 500.0)


def test_sample_3(bayes_net_4):
    torch.manual_seed(703966289599524)
    np.random.seed(703966289599524 % 2**32)

    z = bayes_net_4.sample(1000)
    assert torch.all(z[(z[:, 0] == 0.0) & (z[:, 1] == 0.0), 2:] <= 3.0)
    assert torch.all(z[(z[:, 0] == 0.0) & (z[:, 1] == 0.0), 2:] >= -3.0)

    assert torch.all(z[(z[:, 0] == 0.0) & (z[:, 1] == 1.0), 2:] <= 13.0)
    assert torch.all(z[(z[:, 0] == 0.0) & (z[:, 1] == 1.0), 2:] >= 7.0)

    assert torch.all(z[(z[:, 0] == 1.0) & (z[:, 1] == 0.0), 2:] <= -7.0)
    assert torch.all(z[(z[:, 0] == 1.0) & (z[:, 1] == 0.0), 2:] >= -13.0)

    assert torch.all(z[(z[:, 0] == 1.0) & (z[:, 1] == 1.0), 2:] <= -97.0)
    assert torch.all(z[(z[:, 0] == 1.0) & (z[:, 1] == 1.0), 2:] >= -103.0)


@pytest.mark.parametrize(
    "event,expected_probability",
    [
        (([-3.0, -16.0], [3.0, 16.0]), 1.0),
        (([-3.0, -16.0], [3.0, -14.0]), 0.5),
        (([-3.0, -16.0], [0.0, -14.0]), 0.5),
        (([0.0, -16.0], [3.0, -14.0]), 0.0),
        (([-3.0, 14.0], [3.0, 16.0]), 0.5),
        (([0.0, 14.0], [3.0, 16.0]), 0.5),
        (([-3.0, 14.0], [0.0, 16.0]), 0.0),
        (([-3.0, -15.0], [3.0, 15.0]), 0.5),
        (([0.0, -16.0], [0.0, 16.0]), 0.0),
    ],
)
def test_probability_1(bayes_net_1, event, expected_probability):
    event = tuple(map(torch.tensor, event))
    assert torch.isclose(
        bayes_net_1.probability(event), torch.tensor(expected_probability)
    )


@pytest.mark.parametrize(
    "event,expected_probability",
    [
        (([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]), 1.0),
        (([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0]), 0.3),
        (([0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]), 0.7),
        (([0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]), 0.1 * 0.3 + 0.5 * 0.7),
        (([0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]), 0.9 * 0.3 + 0.5 * 0.7),
        (([1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]), 0.1 * 0.3),
        (([1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]), 0.9 * 0.3),
        (([0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]), 0.5 * 0.7),
        (([0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]), 0.5 * 0.7),
        (([1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]), 0.0),
    ],
)
def test_probability_2(bayes_net_2, event, expected_probability):
    event = tuple(map(torch.tensor, event))
    assert torch.isclose(
        bayes_net_2.probability(event), torch.tensor(expected_probability)
    )


@pytest.mark.parametrize(
    "event,expected_probability",
    [
        (([-3.0, 0.0, 0.0, 0.0, -1.0], [3.0, 1.0, 1.0, 1.0, 1001.0]), 1.0),
        (([-3.0, 1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0, 1001.0]), 0.0),
        (([-3.0, 0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 1.0, 0.0, 1001.0]), 0.25 * 0.5),
        (([-3.0, 0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 1.0, 1001.0]), 0.75 * 0.5),
        (([-3.0, 0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 1.0, 1.0, 1001.0]), 0.5),
        (([0.0, 1.0, 0.0, 0.0, -1.0], [3.0, 1.0, 0.0, 0.0, 1001.0]), 0.9 * 0.5),
        (([0.0, 0.0, 1.0, 0.0, -1.0], [3.0, 0.0, 1.0, 0.0, 1001.0]), 0.1 * 0.5),
        (([0.0, 0.0, 0.0, 1.0, -1.0], [3.0, 0.0, 0.0, 1.0, 1001.0]), 0.0),
        (([0.0, 0.0, 0.0, 0.0, -1.0], [3.0, 1.0, 1.0, 0.0, 1001.0]), 0.5),
        (([0.0, 0.0, 0.0, 0.0, 999.0], [3.0, 1.0, 1.0, 1.0, 1001.0]), 0.0),
    ],
)
def test_probability_3(bayes_net_3, event, expected_probability):
    event = tuple(map(torch.tensor, event))
    assert torch.isclose(
        bayes_net_3.probability(event), torch.tensor(expected_probability)
    )


if __name__ == "__main__":
    pytest.main()
