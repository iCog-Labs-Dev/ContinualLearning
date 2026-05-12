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


if __name__ == "__main__":
    import sys

    sys.path.append(".")
    from core.model import MLP

    key = jax.random.PRNGKey(0)
    model = MLP([4, 8, 8, 3])
    params = model.init_params(key)
    batch_X = jax.random.normal(key, shape=(5, 4))
    pre_acts, post_acts, logits = model.forward_with_states(params, batch_X)

    influence = estimate_influence(params, pre_acts, batch_size=5)
    print("layer_3 w shape:", influence["layer_3"]["w"].shape)  # expect(3, 8)
    print("layer_3 b shape:", influence["layer_3"]["b"].shape)  # expect(3,)
    print("layer_2 w shape:", influence["layer_2"]["w"].shape)  # expect(3, 8)
    print("layer_2 b shape:", influence["layer_2"]["b"].shape)  # expect(3,)
    print("layer_1 w shape:", influence["layer_1"]["w"].shape)  # expect(3, 8)
    print("layer_1 b shape:", influence["layer_1"]["b"].shape)  # expect(3,)
