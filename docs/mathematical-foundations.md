# Mathematical Foundations

## Gini impurity
G(Q) = 1 - sum_k p_k^2

## Entropy
H(Q) = -sum_k p_k log2(p_k), with zero-probability contributions defined as zero.

## Information gain
IG = H(parent) - sum_j (n_j / n) H(child_j)

## Regression mean squared error
MSE(Q) = (1 / n) sum_i (y_i - mean(y))^2

## Cost-complexity pruning
R_alpha(T) = R(T) + alpha * number_of_leaves(T)

Document all conventions and verify formula implementations with hand-calculated examples.
