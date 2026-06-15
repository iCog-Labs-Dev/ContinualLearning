import jax.numpy as jnp

from .activations import relu_derivative


def compute_jacobians(weights, xs_eq):
    jacobians = []
    for l in range(len(weights)):
        avg_mask = jnp.mean(relu_derivative(xs_eq[l]), axis=1)
        jacobians.append(weights[l] * avg_mask[None, :])
    return jacobians


def compute_local_maps(precisions, laterals, jacobians, ridge=1e-4):
    L = len(jacobians)

    def get_lateral(layer_l):
        i = layer_l - 1
        if 0 <= i < len(laterals):
            return laterals[i]
        return None

    def build_H(l):
        H = jnp.diag(precisions[l - 1])
        lat = get_lateral(l)
        if lat is not None:
            H = H + lat

        J_l = jacobians[l]
        pi_l = precisions[l]
        H = H + J_l.T @ (pi_l[:, None] * J_l)
        return H

    G_bar_next = jnp.diag(precisions[L - 1])

    A_tildes = [None] * L
    for l in range(L - 1, -1, -1):
        PJ = precisions[l][:, None] * jacobians[l]
        G_bar_next_reg = G_bar_next + ridge * jnp.eye(G_bar_next.shape[0])
        A_tildes[l] = jnp.linalg.solve(G_bar_next_reg, PJ)

        if l > 0:
            schur = PJ.T @ jnp.linalg.solve(G_bar_next_reg, PJ)
            G_bar_next = build_H(l) - schur

    return A_tildes


def compute_causal_gates(A_tildes, p=2.0, kappa=1e-3):
    gates = []
    for A_k in A_tildes:
        abs_C_p = jnp.abs(A_k) ** p
        denom = jnp.sum(abs_C_p, axis=1, keepdims=True) + kappa
        gates.append(abs_C_p / denom)
    return gates


def compute_composite_influence(A_tildes):
    composites = {}
    C = A_tildes[0]
    composites[1] = C
    for ell in range(1, len(A_tildes)):
        C = C @ A_tildes[ell]
        composites[ell + 1] = C
    return composites
