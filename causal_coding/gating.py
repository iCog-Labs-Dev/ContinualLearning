import os
import sys
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.model import MLP


def estimate_influence(params, pre_acts, batch_size):
    N = len(params)
    W_out = params[f"layer_{N}"]["w"]
    C = W_out.T
    influence = {}
    influence[f"layer_{N}"] = {"w": jnp.abs(C), "b": jnp.ones(W_out.shape[1])}
    C = C[None, :, :]
    num_hidden = len(pre_acts)

    for k in range(num_hidden - 1, -1, -1):
        relu_mask = (pre_acts[k] > 0).astype(float)
        # the wight that produced pre_act[k]
        W = params[f"layer_{k+1}"]["w"]
        J = relu_mask[:, :, None] * W.T[None, :, :]
        C = C @ J

        # store influence for this layer
        influence[f"layer_{k+1}"] = {
            "w": jnp.mean(jnp.abs(C), axis=0),
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


def extract_gate_vectors(params, probe_X, model: MLP, p, kappa):
    pre_acts, _, _ = model.forward_with_states(params, probe_X)
    influence = estimate_influence(params, pre_acts, batch_size=probe_X.shape[0])
    gates = compute_gates(influence, p, kappa)

    gate_vectors = {}

    for layer_key in params:
        gate_w = gates[layer_key]["w"]
        gate_collapsed = jnp.mean(gate_w, axis=0)
        gate_normalized = gate_collapsed / (jnp.mean(gate_collapsed) + 1e-8)
        gate_final = jnp.minimum(gate_normalized, 1.0)

        gate_vectors[layer_key] = gate_final

    return gate_vectors
