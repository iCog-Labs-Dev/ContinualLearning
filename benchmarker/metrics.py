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
        column_i = jnp.array([matrix[t][i] for t in range(i, T)])
        peak = jnp.max(column_i)
        final = matrix[T - 1][i]

        frt_sum += peak - final

    frt = frt_sum / (T - 1)
    return frt


# --- NLL-space continual-learning metrics ---
#
# NLL is lower-is-better, so the sign conventions below are chosen to MATCH
# the accuracy metrics' semantics: positive backward/forward transfer = good,
# positive forgetting = bad. `matrix[t][i]` is the NLL on task i after
# training through task t (0-indexed; T rows).


def backward_transfer_nll(matrix):
    """Diagonal-referenced BWT in NLL space (mirror of `backward_transfer`).

    mean_i ( N_{i,i} − N_{T,i} ) over i < T. Positive = subsequent training
    LOWERED the NLL on an earlier task (improvement). This is the sign-flipped
    analogue of the accuracy BWT (final − diagonal).
    """
    T = len(matrix)
    total = 0
    for i in range(T - 1):
        diagonal_value = matrix[i][i]
        final_value = matrix[T - 1][i]
        total += diagonal_value - final_value
    return total / (T - 1)


def forgetting_nll(matrix):
    """Best-referenced forgetting in NLL space (mirror of `forgetting`).

    mean_i ( N_{T,i} − min_{t≥i} N_{t,i} ) over i < T. Positive = the NLL on an
    earlier task rose above its best (minimum) value = forgetting. Uses the
    BEST/MINIMUM NLL as the reference, never a "peak".
    """
    T = len(matrix)
    total = 0
    for i in range(T - 1):
        column_i = jnp.array([matrix[t][i] for t in range(i, T)])
        best = jnp.min(column_i)
        final = matrix[T - 1][i]
        total += final - best
    return total / (T - 1)


def bwt_nll(matrix):
    """Best-referenced BWT in NLL space: BWT_NLL = −forgetting_nll.

    mean_i ( min_{t≥i} N_{t,i} − N_{T,i} ). Positive = final NLL is below the
    best seen after task i (net improvement). Reported alongside the
    diagonal-referenced `backward_transfer_nll`.
    """
    return -forgetting_nll(matrix)


def forward_transfer_nll(matrix, baselines):
    """FWT in NLL space (mirror of `forward_transfer`).

    mean_{i≥1} ( N_baseline,i − N_{i-1,i} ). Positive = prior training LOWERED
    the NLL on a not-yet-seen task relative to the untrained-model baseline
    (helpful forward transfer). `baselines[i]` is the untrained model's NLL on
    task i.
    """
    T = len(matrix)
    total = 0
    for i in range(1, T):
        before_training = matrix[i - 1][i]
        baseline = baselines[i]
        total += baseline - before_training
    return total / (T - 1)
