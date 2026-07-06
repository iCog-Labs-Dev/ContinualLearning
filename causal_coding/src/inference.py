import jax
import jax.numpy as jnp

from causal_coding.src.activations import relu, relu_derivative


def compute_errors(weights, xs, task_il_training=False, active_mask=None):
    """Top-down prediction errors at every layer.

    The output edge (``l == num_layers - 1``) depends on the protocol:

    - Class-IL (default): categorical softmax head, ``error = xs[L] − softmax(pred)``.
    - Task-IL (``task_il_training=True``): independent-Bernoulli head over the
      task's active classes, ``error = active_mask · (xs[L] − sigmoid(pred))``.
      ``active_mask`` is a ``(num_classes,)`` 0/1 array; it zeros the residual
      on the output units of classes outside the current task, so inactive
      sigmoid heads carry no learning signal (no pseudo-loss).

    Hidden layers are Gaussian residuals regardless of protocol.
    """
    num_layers = len(xs)
    errors = [jnp.zeros_like(xs[0])]
    for l in range(1, num_layers):
        prediction = weights[l - 1] @ relu(xs[l - 1])
        if l == num_layers - 1:
            if task_il_training:
                probs = jax.nn.sigmoid(prediction)
                error = active_mask[:, None] * (xs[l] - probs)
            else:
                probs = jax.nn.softmax(prediction, axis=0)
                error = xs[l] - probs
        else:
            error = xs[l] - prediction
        errors.append(error)

    return errors


def _infer_step(
    weights, xs, precisions, lateral_pairs, lr_z,
    task_il_training=False, active_mask=None,
):
    """One PC inference step with low-rank lateral force.

    `lateral_pairs` is a list of `(alpha_eff, U)` tuples — one per HIDDEN
    layer (length = num_layers - 2, NOT num_layers - 1). The output edge
    has no lateral. Hidden layer index l (in the iteration over
    xs[1..num_layers-2]) picks `lateral_pairs[l-1]`.

    `task_il_training` / `active_mask` select the output-edge error model
    (softmax vs masked sigmoid); see `compute_errors`.
    """
    num_layers = len(xs)
    expected_pairs = num_layers - 2
    assert len(lateral_pairs) == expected_pairs, (
        f"_infer_step: lateral_pairs has {len(lateral_pairs)} entries; "
        f"expected {expected_pairs} (hidden layers only, output edge "
        f"excluded)."
    )

    errors = compute_errors(weights, xs, task_il_training, active_mask)
    new_xs = [xs[0]]
    for l in range(1, num_layers - 1):
        weighted_above = precisions[l][:, None] * errors[l + 1]
        weighted_self = precisions[l - 1][:, None] * errors[l]
        top_down = weights[l].T @ weighted_above * relu_derivative(xs[l])
        # Low-rank lateral: α · U (Uᵀ z), O(d·r·B) vs O(d²·B).
        alpha_eff, U = lateral_pairs[l - 1]
        lateral = alpha_eff * (U @ (U.T @ xs[l]))
        new_xs.append(xs[l] + lr_z * (top_down - weighted_self - lateral))

    new_xs.append(xs[-1])
    return new_xs


def infer(
    weights, xs, precisions, lateral_pairs, num_steps, lr_z,
    task_il_training=False, active_mask=None,
):
    """Run `num_steps` PC inference steps from initial `xs`.

    `lateral_pairs`: list of `(alpha_eff, U)` tuples for hidden layers.
    `task_il_training` / `active_mask` select the output-edge error model.
    """
    def body_fn(_, xs):
        return _infer_step(
            weights, xs, precisions, lateral_pairs, lr_z,
            task_il_training, active_mask,
        )

    return jax.lax.fori_loop(0, num_steps, body_fn, xs)


def infer_with_trajectory(
    weights, xs, precisions, lateral_pairs, num_steps, lr_z,
    task_il_training=False, active_mask=None,
):
    """Run PC inference and return the full state trajectory.

    Returns `(xs_eq, trajectory)`, where `xs_eq` matches the final state
    returned by `infer` and `trajectory[l]` has shape
    `(d_l, batch, num_steps + 1)`, including the initial state at `t=0`.

    Used by regression diagnostics, not by the training loop.
    """
    trajectory = [jnp.zeros(z.shape + (num_steps + 1,), dtype=z.dtype) for z in xs]
    trajectory = [
        traj_l.at[..., 0].set(z_l)
        for traj_l, z_l in zip(trajectory, xs)
    ]

    cur = xs
    for t in range(num_steps):
        cur = _infer_step(
            weights, cur, precisions, lateral_pairs, lr_z,
            task_il_training, active_mask,
        )
        trajectory = [
            traj_l.at[..., t + 1].set(z_l)
            for traj_l, z_l in zip(trajectory, cur)
        ]

    return cur, trajectory
