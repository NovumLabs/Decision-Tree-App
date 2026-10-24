"""Candidate split generation and evaluation."""

from dataclasses import dataclass
from typing import Any

from decision_tree_ai.criteria.classification import entropy


@dataclass
class SplitResult:
    """Store the result of a candidate split."""

    feature: str
    information_gain: float
    groups: dict[Any, list[int]]


def best_split(
    X: list[dict[str, Any]],
    y: list[Any],
) -> SplitResult | None:
    """Find the categorical feature with the highest information gain."""

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if not X:
        raise ValueError("The dataset cannot be empty.")

    if not X[0]:
        return None

    feature_names = set(X[0].keys())

    if any(set(row.keys()) != feature_names for row in X):
        raise ValueError("Every row must contain the same features.")

    if any(value is None for row in X for value in row.values()):
        raise ValueError("Missing feature values are not supported yet.")

    if any(label is None for label in y):
        raise ValueError("Missing class labels are not supported yet.")

    parent_counts = {}

    for label in y:
        parent_counts[label] = parent_counts.get(label, 0) + 1

    parent_probabilities = [
        count / len(y) for count in parent_counts.values()
    ]

    parent_entropy = entropy(parent_probabilities)
    best_result = None

    for feature in X[0]:
        groups = {}

        for index, row in enumerate(X):
            value = row[feature]
            groups.setdefault(value, []).append(index)

        if len(groups) < 2:
            continue

        weighted_entropy = 0.0

        for indices in groups.values():
            counts = {}

            for index in indices:
                label = y[index]
                counts[label] = counts.get(label, 0) + 1

            probabilities = [
                count / len(indices) for count in counts.values()
            ]

            weighted_entropy += (
                len(indices) / len(y)
            ) * entropy(probabilities)

        gain = parent_entropy - weighted_entropy

        if (
            best_result is None
            or gain > best_result.information_gain
        ):
            best_result = SplitResult(
                feature=feature,
                information_gain=gain,
                groups=groups,
            )

    return best_result
