"""VLCP sparse-regression diagnostic helpers.

This module is diagnostic-only. It fits one LASSO per child neuron over
inference trajectories:

    Delta z_child,j(t) ~= sum_i a[j, i] * z_parent,i(t)

and returns d_reg[j, i] = |a[j, i]| as the regression-based dynamic
importance score.
"""

import jax
import jax.numpy as jnp


def _soft_threshold(x, lam):
    return jnp.sign(x) * jnp.maximum(jnp.abs(x) - lam, 0.0)


def _center_matrix(X):
    return X - jnp.mean(X, axis=0, keepdims=True)


def _prepare_predictors(X, standardize_x=False):
    X_centered = _center_matrix(X)
    if not standardize_x:
        return X_centered

    col_std = jnp.std(X_centered, axis=0, keepdims=True)
    col_std = jnp.where(col_std > 1e-12, col_std, 1.0)
    return X_centered / col_std


def _lasso_cd_one_child_centered(X, y, lambda_dyn, num_iters):
    """Coordinate-descent LASSO for centered X and centered y."""
    n_features = X.shape[1]
    col_norm_sq = jnp.sum(X ** 2, axis=0) + 1e-12
    w0 = jnp.zeros((n_features,), dtype=X.dtype)
    residual0 = y

    def coord_body(i, state):
        w, residual = state
        x_i = X[:, i]
        old_w_i = w[i]
        residual_without_i = residual + x_i * old_w_i
        rho = jnp.dot(x_i, residual_without_i)
        new_w_i = _soft_threshold(rho, lambda_dyn) / col_norm_sq[i]
        residual = residual_without_i - x_i * new_w_i
        w = w.at[i].set(new_w_i)
        return w, residual

    def iter_body(_, state):
        return jax.lax.fori_loop(0, n_features, coord_body, state)

    w, _residual = jax.lax.fori_loop(
        0, num_iters, iter_body, (w0, residual0)
    )
    return w


def lasso_cd_one_child(
    X,
    y,
    lambda_dyn=1e-3,
    num_iters=50,
    standardize_x=False,
):
    """Fit one per-child LASSO regression.

    Args:
        X: Predictor matrix, shape `(num_samples, parent_dim)`.
        y: Target vector, shape `(num_samples,)`.
        lambda_dyn: L1 penalty for the dynamic regression.
        num_iters: Number of coordinate-descent sweeps.
        standardize_x: If True, center and scale each predictor column before
            fitting. Coefficients are then in standardized-predictor units.

    Returns:
        Coefficients with shape `(parent_dim,)`.
    """
    X_centered = _prepare_predictors(X, standardize_x)
    y_centered = y - jnp.mean(y)
    return _lasso_cd_one_child_centered(
        X_centered, y_centered, lambda_dyn, num_iters
    )


def lasso_cd_layer(
    X,
    Y,
    lambda_dyn=1e-3,
    num_iters=50,
    standardize_x=False,
):
    """Fit independent per-child LASSO regressions for one layer pair.

    Args:
        X: Predictor matrix, shape `(num_samples, parent_dim)`.
        Y: Target matrix, shape `(num_samples, child_dim)`.
        standardize_x: If True, center and scale each predictor column before
            fitting. Coefficients are then in standardized-predictor units.

    Returns:
        Coefficients with shape `(child_dim, parent_dim)`.
    """
    X_centered = _prepare_predictors(X, standardize_x)
    Y_centered = _center_matrix(Y)
    return jax.vmap(
        lambda y: _lasso_cd_one_child_centered(
            X_centered, y, lambda_dyn, num_iters
        ),
        in_axes=1,
        out_axes=0,
    )(Y_centered)


def _resolve_window(num_steps, window, window_start, window_end):
    """Resolve last-N or explicit half-open windows over inference steps."""
    num_steps = int(num_steps)
    if num_steps < 1:
        raise ValueError("VLCP regression needs at least one inference step.")

    if window_start is None and window_end is None:
        window_steps = num_steps if window is None else min(int(window), num_steps)
        start = num_steps - window_steps
        end = num_steps
    else:
        start = 0 if window_start is None else int(window_start)
        end = num_steps if window_end is None else int(window_end)
        start = min(max(start, 0), num_steps)
        end = min(max(end, 0), num_steps)

        if end <= start:
            end = max(1, end)
            end = min(end, num_steps)
            start = max(0, end - 1)

    return int(start), int(end)


def _trajectory_window(
    parent_trajectory,
    child_trajectory,
    window=15,
    window_start=None,
    window_end=None,
):
    """Assemble predictors and one-step targets from trajectories."""
    num_steps = parent_trajectory.shape[-1] - 1
    start, end = _resolve_window(num_steps, window, window_start, window_end)

    parent_states = parent_trajectory[..., start:end]
    child_delta = (
        child_trajectory[..., start + 1:end + 1]
        - child_trajectory[..., start:end]
    )

    # (d, batch, time) -> (time * batch, d)
    X = jnp.transpose(parent_states, (2, 1, 0)).reshape(
        (-1, parent_trajectory.shape[0])
    )
    Y = jnp.transpose(child_delta, (2, 1, 0)).reshape(
        (-1, child_trajectory.shape[0])
    )
    return X, Y, {
        "window_start": int(start),
        "window_end": int(end),
        "window_steps": int(end - start),
    }


def compute_dreg_for_pair(
    trajectory_parent,
    trajectory_child,
    window=15,
    window_start=None,
    window_end=None,
    lambda_dyn=1e-3,
    num_iters=50,
    standardize_x=False,
):
    """Compute d_reg for one parent-child layer pair.

    `trajectory_parent` and `trajectory_child` have shapes
    `(d_parent, batch, num_steps + 1)` and
    `(d_child, batch, num_steps + 1)`. The predictor is the parent state
    z_l(t); the target is the one-step child change Delta z_{l+1}(t).

    Returns `(d_reg, window_info)` where `d_reg` has shape
    `(d_child, d_parent)`.
    """
    X, Y, window_info = _trajectory_window(
        trajectory_parent,
        trajectory_child,
        window=window,
        window_start=window_start,
        window_end=window_end,
    )
    coeffs = lasso_cd_layer(
        X,
        Y,
        lambda_dyn,
        num_iters,
        standardize_x=standardize_x,
    )
    window_info["standardize_x"] = bool(standardize_x)
    return jnp.abs(coeffs), window_info
