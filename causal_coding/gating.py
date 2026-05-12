import jax
import jax.numpy as jnp


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
