
import math
import pytest

from decision_tree_ai.criteria.classification import (
    entropy,
    gini_impurity,
)


def test_entropy_equal_classes():
    assert entropy([0.5, 0.5]) == 1.0


def test_entropy_pure_class():
    assert entropy([1.0, 0.0]) == 0.0


def test_entropy_unequal_classes():
    assert math.isclose(
        entropy([0.75, 0.25]),
        0.8112781244591328,
    )


def test_entropy_invalid_probabilities():
    with pytest.raises(ValueError):
        entropy([0.7, 0.5])


def test_gini_equal_classes():
    assert math.isclose(gini_impurity([0.5, 0.5]), 0.5)


def test_gini_pure_class():
    assert gini_impurity([1.0, 0.0]) == 0.0


def test_gini_unequal_classes():
    assert math.isclose(
        gini_impurity([0.75, 0.25]),
        0.375,
    )


def test_gini_invalid_probabilities():
    with pytest.raises(ValueError):
        gini_impurity([0.7, 0.5])



def test_entropy_single_class():
    assert entropy([1.0]) == 0.0


def test_entropy_ignores_zero_probability():
    assert entropy([1.0, 0.0]) == 0.0


def test_entropy_rejects_empty_probabilities():
    with pytest.raises(ValueError, match="cannot be empty"):
        entropy([])


def test_entropy_rejects_negative_probability():
    with pytest.raises(ValueError, match="between 0 and 1"):
        entropy([1.1, -0.1])


def test_gini_single_class():
    assert gini_impurity([1.0]) == 0.0


def test_gini_rejects_empty_probabilities():
    with pytest.raises(ValueError, match="cannot be empty"):
        gini_impurity([])


def test_gini_rejects_negative_probability():
    with pytest.raises(ValueError, match="between 0 and 1"):
        gini_impurity([1.1, -0.1])


def test_gini_rejects_probabilities_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        gini_impurity([0.2, 0.2])
