from decision_tree_ai.criteria.classification import entropy


def information_gain(parent_probabilities, child_groups):
    """
    Calculate information gain from splitting a parent group.

    parent_probabilities: class probabilities before splitting
    child_groups: list of (group_size, class_probabilities) tuples
    """
    if not child_groups:
        raise ValueError("Child groups cannot be empty.")

    if any(size <= 0 for size, _ in child_groups):
        raise ValueError("Group sizes must be positive.")

    total_size = sum(size for size, _ in child_groups)

    parent_entropy = entropy(parent_probabilities)

    weighted_child_entropy = sum(
        (size / total_size) * entropy(probabilities)
        for size, probabilities in child_groups
    )

    return parent_entropy - weighted_child_entropy