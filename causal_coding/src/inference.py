import jax
import jax.numpy as jnp

from causal_coding.src.activations import relu, relu_derivative


def compute_errors(weights, xs):
    num_layers = len(xs)
    errors = [jnp.zeros_like(xs[0])]
    for l in range(1, num_layers):
        prediction = weights[l - 1] @ relu(xs[l - 1])
        if l == num_layers - 1:
            probs = jax.nn.softmax(prediction, axis=0)
            error = xs[l] - probs
        else:
            error = xs[l] - prediction
        errors.append(error)

    return errors


def _infer_step(weights, xs, precisions, lateral_pairs, lr_z):
    """One PC inference step with low-rank lateral force.

    `lateral_pairs` is a list of `(alpha_eff, U)` tuples — one per HIDDEN
    layer (length = num_layers - 2, NOT num_layers - 1). The output edge
    has no lateral. Hidden layer index l (in the iteration over
    xs[1..num_layers-2]) picks `lateral_pairs[l-1]`.
    """
    num_layers = len(xs)
    expected_pairs = num_layers - 2
    assert len(lateral_pairs) == expected_pairs, (
        f"_infer_step: lateral_pairs has {len(lateral_pairs)} entries; "
        f"expected {expected_pairs} (hidden layers only, output edge "
        f"excluded)."
    )

    errors = compute_errors(weights, xs)
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


def infer(weights, xs, precisions, lateral_pairs, num_steps, lr_z):
    """Run `num_steps` PC inference steps from initial `xs`.

    `lateral_pairs`: list of `(alpha_eff, U)` tuples for hidden layers.
    """
    def body_fn(_, xs):
        return _infer_step(weights, xs, precisions, lateral_pairs, lr_z)

    return jax.lax.fori_loop(0, num_steps, body_fn, xs)
