import jax
import jax.numpy as jnp

from causal_coding.src.activations import relu, relu_derivative


def compute_jacobians(weights, xs_eq):
    jacobians = []
    for l in range(len(weights)):
        avg_mask = jnp.mean(relu_derivative(xs_eq[l]), axis=1)
        jacobians.append(weights[l] * avg_mask[None, :])
    return jacobians


def compute_output_fisher(weights, xs_eq):
    """Categorical Fisher F_cat = mean_b [diag(p_b) − p_b p_bᵀ] at equilibrium.

    Used as the curvature at the output Schur base case (discriminative
    classification head). Replaces the Gaussian `diag(π_L)` everywhere the
    output edge enters the Schur recursion.
    """
    logits = weights[-1] @ relu(xs_eq[-2])
    probs = jax.nn.softmax(logits, axis=0)
    B = probs.shape[1]
    return jnp.diag(jnp.mean(probs, axis=1)) - (probs @ probs.T) / B


def compute_local_maps(precisions, lateral_pairs, jacobians, ridge=1e-4, output_fisher=None):
    """Schur–Fisher backward recursion for the per-layer local natural maps.

    `output_fisher` is the optional categorical Fisher matrix at the output
    edge. When provided, it replaces the Gaussian `diag(π_L)` in three
    places: the Schur base case `G_bar_L`, the output cross-term
    `PJ_{L-1} = F · J_{L-1}`, and the output's contribution to `H_{L-1}`
    via `J_{L-1}ᵀ · F · J_{L-1}`. Hidden layers keep the Gaussian diagonal
    machinery. If omitted, the output edge uses its diagonal precision.

    `lateral_pairs` is the low-rank lateral state: a list of
    `(alpha_eff, U)` tuples, one per hidden layer. The output edge has no
    lateral. Λ_l = α_l · U_l Uᵀ_l is materialised on demand to a dense
    (d, d) matrix inside `get_lateral`; the Schur recursion needs the dense
    form for H_l = Π + Λ + JᵀΠJ.
    """
    L = len(jacobians)
    # Shape check: lateral_pairs has one entry per hidden layer.
    expected_pairs = L - 1
    assert len(lateral_pairs) == expected_pairs, (
        f"compute_local_maps: lateral_pairs has {len(lateral_pairs)} "
        f"entries; expected {expected_pairs} (hidden layers only, output "
        f"edge excluded)."
    )
    for i, pair in enumerate(lateral_pairs):
        alpha, U = pair
        assert U.ndim == 2, f"lateral_pairs[{i}] U must be 2D, got shape {U.shape}"

    def get_lateral(layer_l):
        # layer_l ∈ {1..L} indexes xs[l] inside build_H. lateral_pairs[i] for
        # i = layer_l - 1 covers hidden layers xs[1..L-1]; xs[L] (output) gets
        # no lateral and returns None.
        i = layer_l - 1
        if 0 <= i < len(lateral_pairs):
            alpha, U = lateral_pairs[i]
            return alpha * (U @ U.T)
        return None

    def apply_above_curvature(l, X):
        """Left-multiply X by the curvature of the layer above (l+1).

        At the output edge with `output_fisher` supplied, this is a full
        matrix multiply by F_cat. Hidden layers use the cheap element-wise
        diagonal form (`π[:, None] * X`).
        """
        if l == L - 1 and output_fisher is not None:
            return output_fisher @ X
        return precisions[l][:, None] * X

    def build_H(l):
        H = jnp.diag(precisions[l - 1])
        lat = get_lateral(l)
        if lat is not None:
            H = H + lat
        J_l = jacobians[l]
        H = H + J_l.T @ apply_above_curvature(l, J_l)
        return H

    if output_fisher is not None:
        G_bar_next = output_fisher
    else:
        G_bar_next = jnp.diag(precisions[L - 1])
    out_lateral = get_lateral(L)
    if out_lateral is not None:
        G_bar_next = G_bar_next + out_lateral

    A_tildes = [None] * L
    for l in range(L - 1, -1, -1):
        PJ = apply_above_curvature(l, jacobians[l])
        G_bar_next_reg = G_bar_next + ridge * jnp.eye(G_bar_next.shape[0])
        A_tildes[l] = jnp.linalg.solve(G_bar_next_reg, PJ)

        if l > 0:
            schur = PJ.T @ jnp.linalg.solve(G_bar_next_reg, PJ)
            G_bar_next = build_H(l) - schur

    return A_tildes


def compute_causal_gates(A_tildes, p=2.0, kappa=1e-3, alpha=0.6, floor_coeff=0.05):
    """Amplitude-preserving causal gate.

    Each row's R = |A|^p / Σ|A|^p (row-sum ≈ 1) is rescaled into a gate
    whose row-sum is ≈ n_in, so the mean gate value is 1 instead of 1/n_in:

        G[i, j] = max( 1 + α · (n_in · R[i, j] − 1),  1 − α + floor_coeff · α )

    α = 0 → identity gate (no causal selectivity, plain Hebbian). Larger α
    amplifies high-importance synapses and softly damps low-importance
    synapses toward 1 − α.
    """
    gates = []
    for A_k in A_tildes:
        abs_C_p = jnp.abs(A_k) ** p
        denom = jnp.sum(abs_C_p, axis=1, keepdims=True) + kappa
        R = abs_C_p / denom
        n_in = A_k.shape[1]
        G = 1.0 + alpha * (n_in * R - 1.0)
        floor = 1.0 - alpha + floor_coeff * alpha
        gates.append(jnp.maximum(G, floor))
    return gates


def compute_composite_influence(A_tildes):
    composites = {}
    C = A_tildes[0]
    composites[1] = C
    for ell in range(1, len(A_tildes)):
        C = C @ A_tildes[ell]
        composites[ell + 1] = C
    return composites
