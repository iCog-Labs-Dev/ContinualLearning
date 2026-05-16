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

    return jnp.mean(jnp.array(per_layer)), per_layer


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

    return jnp.mean(jnp.array(per_layer)), per_layer


def clarity_penalty(current_gates, all_prev_gates):
    if all_prev_gates == []:
        return 0.0
    total = 0.0

    for prev_gates in all_prev_gates:
        layers_sims = []

        for layer_key in current_gates:
            vi = current_gates[layer_key]
            vj = prev_gates[layer_key]

            dot = jnp.sum(vi * vj)
            norm_i = jnp.sqrt(jnp.sum(vi * vi))
            norm_j = jnp.sqrt(jnp.sum(vj * vj))

            cosine = dot / (norm_i * norm_j + 1e-8)

            layers_sims.append(cosine)
        total += jnp.mean(jnp.array(layers_sims))

    return total / len(all_prev_gates)
