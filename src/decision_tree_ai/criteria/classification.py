"""Gini and entropy criteria for classification."""

import math


def entropy(probabilities):
    """Calculate entropy for a probability distribution."""
    if not probabilities:
        raise ValueError("Probabilities cannot be empty.")

    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("Each probability must be between 0 and 1.")

    if not math.isclose(sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum to 1.")

    return -sum(
        p * math.log2(p)
        for p in probabilities
        if p > 0
    )


def gini_impurity(probabilities):
    """Calculate Gini impurity for a probability distribution."""
    if not probabilities:
        raise ValueError("Probabilities cannot be empty.")

    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("Each probability must be between 0 and 1.")

    if not math.isclose(sum(probabilities), 1.0):
        raise ValueError("Probabilities must sum to 1.")

    return 1.0 - sum(p ** 2 for p in probabilities)
