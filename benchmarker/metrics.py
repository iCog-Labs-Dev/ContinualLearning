import jax.numpy as jnp


def average_accuracy(matrix):
    return jnp.mean(jnp.array(matrix)[-1])


def backward_transfer(matrix):
    T = len(matrix)

    bwt_sum = 0
    for i in range(T - 1):
        diagonal_value = matrix[i][i]
        final_value = matrix[T - 1][i]

        drop = final_value - diagonal_value
        bwt_sum += drop

    bwt = bwt_sum / (T - 1)
    return bwt


def forward_transfer(matrix, baselines):
    T = len(matrix)
    fwt_sum = 0

    for i in range(1, T):
        before_training = matrix[i - 1][i]
        baseline = baselines[i]
        fwt_sum += before_training - baseline

    fwt = fwt_sum / (T - 1)
    return fwt


def forgetting(matrix):
    T = len(matrix)
    frt_sum = 0

    for i in range(T - 1):
        column_i = jnp.array(matrix[t][i] for t in range(i, T))
        peak = jnp.max(column_i)
        final = matrix[T - 1][i]

        frt_sum += peak - final

    frt = frt_sum / (T - 1)
    return frt
