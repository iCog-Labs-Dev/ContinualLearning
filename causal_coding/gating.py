import os
import sys
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.model import MLP


def estimate_influence(params, pre_acts, influence_mode="composite"):
    N = len(params)
    W_out = params[f"layer_{N}"]["w"]
    influence = {}

    if influence_mode == "composite":
        C = W_out.T
        influence[f"layer_{N}"] = {"w": jnp.abs(C), "b": jnp.ones(W_out.shape[1])}
        C = C[None, :, :]
        num_hidden = len(pre_acts)

        for k in range(num_hidden - 1, -1, -1):
            relu_mask = (pre_acts[k] > 0).astype(float)
            # the weight that produced pre_act[k]
            W = params[f"layer_{k+1}"]["w"]
            J = relu_mask[:, :, None] * W.T[None, :, :]
            C = C @ J

            # store influence for this layer
            influence[f"layer_{k+1}"] = {
                "w": jnp.mean(jnp.abs(C), axis=0),
                "b": jnp.ones(W.shape[1]),
            }

    elif influence_mode == "local":
        # Output layer: local map is W_out^T (linear, no ReLU)
        influence[f"layer_{N}"] = {"w": jnp.abs(W_out.T), "b": jnp.ones(W_out.shape[1])}

        num_hidden = len(pre_acts)
        for k in range(num_hidden - 1, -1, -1):
            relu_mask = (pre_acts[k] > 0).astype(float)
            W = params[f"layer_{k+1}"]["w"]
            # A_k = diag(relu_mask) @ W^T, per sample
            A_k = relu_mask[:, :, None] * W.T[None, :, :]
            # Mean over batch, absolute value -> [d_out, d_in] edge-aligned with W_k
            influence[f"layer_{k+1}"] = {
                "w": jnp.mean(jnp.abs(A_k), axis=0),
                "b": jnp.ones(W.shape[1]),
            }

    return influence


def compute_gates(influence, p, kappa):
    gates = {}
    for layer_key, layer_influence in influence.items():
        infl_w = layer_influence["w"]
        numerator = infl_w**p
        denominator = jnp.sum(numerator, axis=1, keepdims=True) + kappa
        gate_w = numerator / denominator
        gate_b = jnp.ones_like(layer_influence["b"])

        gates[layer_key] = {"w": gate_w, "b": gate_b}

    return gates


def _percentile_normalize(gate, gate_quantile):
    ref = jnp.percentile(gate, gate_quantile * 100)
    return jnp.clip(gate / (ref + 1e-8), 0.0, 1.0)


def extract_gate_vectors(params, probe_X, model: MLP, p, kappa, gate_quantile=0.90,
                         influence_mode="composite"):
    pre_acts, _, _ = model.forward_with_states(params, probe_X)
    influence = estimate_influence(params, pre_acts, influence_mode=influence_mode)
    gates = compute_gates(influence, p, kappa)

    gate_vectors = {}

    for layer_key in params:
        gate_w = gates[layer_key]["w"]
        if influence_mode == "local" and gate_w.ndim == 2:
            # local mode: edge matrix [d_out, d_in] — collapse to per-input-unit
            gate_collapsed = jnp.max(gate_w, axis=0)
        else:
            gate_collapsed = jnp.max(gate_w, axis=0)
        gate_final = _percentile_normalize(gate_collapsed, gate_quantile)

        gate_vectors[layer_key] = gate_final

    return gate_vectors


def compute_support_mask(gate_vectors, support_frac=0.15):
    support_mask = {}
    for layer_key in gate_vectors:
        gate_vec = gate_vectors[layer_key]
        k = int(jnp.ceil(support_frac * gate_vec.shape[0]))
        # k-th largest value as threshold
        sorted_vals = jnp.sort(gate_vec)
        threshold = sorted_vals[-k]
        support_mask[layer_key] = (gate_vec >= threshold).astype(float)

    return support_mask
