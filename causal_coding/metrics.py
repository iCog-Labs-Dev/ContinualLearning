import jax
import jax.numpy as jnp


def gate_jaccard(g_i, g_j, threshold=0.5):
    per_layer = []

    for layer_key in g_i:
        s_i = g_i[layer_key] > threshold
        s_j = g_j[layer_key] > threshold

        intersection = jnp.sum(s_i & s_j)
        union = jnp.sum(s_i | s_j)

        if union == 0:
            j = 1.0
        else:
            j = intersection / union

        per_layer.append(j)

    return jnp.mean(per_layer), per_layer


def commutator_proxy(g_i, g_j):
    per_layer = []

    for layer_key in g_i:
        vi = g_i[layer_key]
        vj = g_j[layer_key]

        dot = jnp.sum(vi * vj)
        norm_i = jnp.sqrt(jnp.sum(vi * vi))
        norm_j = jnp.sqrt(jnp.sum(vj * vj))

        cosine = dot / (norm_i * norm_j + 1e-8)
        distance = 1.0 - cosine

        per_layer.append(distance)

    return jnp.mean(per_layer), per_layer
