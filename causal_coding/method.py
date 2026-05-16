import os
import sys
import jax
import jax.numpy as jnp
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.model import MLP
from core.metrics import cross_entropy
from causal_coding.gating import (
    estimate_influence,
    compute_gates,
    extract_gate_vectors,
    compute_support_mask,
    _percentile_normalize,
)
from core.data import Task
from core.base import CausalState

from causal_coding.metrics import clarity_penalty


def _loss_fn(params, X, y, model: MLP):
    logits = model.forward(params, X)
    return cross_entropy(logits, y)


@partial(jax.jit, static_argnums=(6, 10, 12, 13))
def _gated_train_step(
    params,
    X,
    y,
    lr,
    p,
    kappa,
    model: MLP,
    lambda_clarity,
    all_prev_gate_vecs,
    accumulated_support,
    influence_mode,
    gate_quantile,
    seen_classes,
    use_head_protection,
):
    loss, grads = jax.value_and_grad(_loss_fn)(params, X, y, model)
    pre_acts, _, _ = model.forward_with_states(params, X)
    # compute the influence and gates
    influence = estimate_influence(params, pre_acts, influence_mode=influence_mode)
    gates = compute_gates(influence, p, kappa)
    gated_grads = {}
    final_gates = []

    N = len(params)
    output_layer_key = "layer_" + str(N)

    for layer_key in params:
        gate_w = gates[layer_key]["w"]
        if layer_key == output_layer_key:
            # Output layer: edge-wise gating
            gate_edge = gate_w.T
            gate_edge_sel = _percentile_normalize(gate_edge, gate_quantile)

            gated_grad_w = grads[layer_key]["w"] * gate_edge_sel
            # Scalar LR restore
            gated_grad_w = gated_grad_w / (jnp.mean(gate_edge_sel) + 1e-8)

            gate_for_tracking = jnp.max(gate_w, axis=0)
            gate_for_tracking = _percentile_normalize(gate_for_tracking, gate_quantile)
        else:
            if influence_mode == "local":
                # local mode: gate_w is [d_out, d_in], W_k is [d_in, d_out]
                # Transpose gate to align with weight shape
                gate_edge_sel = _percentile_normalize(gate_w.T, gate_quantile)
                gated_grad_w = grads[layer_key]["w"] * gate_edge_sel
                # Scalar LR restore
                gated_grad_w = gated_grad_w / (jnp.mean(gate_edge_sel) + 1e-8)
                # For tracking: collapse to per-input-unit (axis=1 = d_out)
                gate_for_tracking = jnp.max(gate_edge_sel, axis=1)
            else:
                # composite mode: gate_w has class dim, max-collapse
                gate_collapsed = jnp.max(gate_w, axis=0)
                gate_sel = _percentile_normalize(gate_collapsed, gate_quantile)
                gate_matrix = gate_sel[:, None]
                gated_grad_w = grads[layer_key]["w"] * gate_matrix
                # Scalar LR restore
                gated_grad_w = gated_grad_w / (jnp.mean(gate_sel) + 1e-8)
                gate_for_tracking = gate_sel

        gated_grads[layer_key] = {
            "w": gated_grad_w,
            "b": grads[layer_key]["b"] * gates[layer_key]["b"],
        }

        final_gates.append(jnp.ravel(gate_for_tracking))

    # Support protection: zero out gradients for protected units
    if len(accumulated_support) > 0:
        for layer_key in params:
            protection_vec = accumulated_support[layer_key]
            allow_matrix = (1.0 - protection_vec)[:, None]
            gated_grads[layer_key]["w"] = gated_grads[layer_key]["w"] * allow_matrix

    # Output-head protection: zero gradient columns for already-seen classes
    if use_head_protection and len(seen_classes) > 0:
        head_mask = jnp.ones_like(gated_grads[output_layer_key]["w"])
        head_mask = head_mask.at[:, jnp.array(seen_classes)].set(0.0)
        gated_grads[output_layer_key]["w"] = gated_grads[output_layer_key]["w"] * head_mask
        # Also zero bias for seen classes
        bias_mask = jnp.ones_like(gated_grads[output_layer_key]["b"])
        bias_mask = bias_mask.at[jnp.array(seen_classes)].set(0.0)
        gated_grads[output_layer_key]["b"] = gated_grads[output_layer_key]["b"] * bias_mask

    # Clarity penalty
    if len(all_prev_gate_vecs) > 0:

        def clarity_fn(theta):
            pre, _, _ = model.forward_with_states(theta, X)
            infl = estimate_influence(theta, pre, influence_mode=influence_mode)
            g = compute_gates(infl, p, kappa)
            cvecs = {}
            for lk in theta:
                gw = g[lk]["w"]
                gc = jnp.max(gw, axis=0)
                gf = _percentile_normalize(gc, gate_quantile)
                cvecs[lk] = gf

            return clarity_penalty(cvecs, all_prev_gate_vecs)

        clarity_grads = jax.grad(clarity_fn)(params)

    else:
        clarity_grads = jax.tree.map(jnp.zeros_like, params)

    new_params = jax.tree.map(
        lambda param, g, cg: param - lr * (g + lambda_clarity * cg),
        params,
        gated_grads,
        clarity_grads,
    )

    all_gates = jnp.concatenate(final_gates)

    sparsity = jnp.mean(all_gates < 0.5)

    return new_params, loss, sparsity


class CausalMethod:
    def __init__(
        self, lr, batch_size, epochs, p, kappa, lambda_clarity, use_protection,
        gate_quantile=0.90, support_frac=0.15, influence_mode="composite",
        use_head_protection=False,
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.p = p
        self.kappa = kappa
        self.lambda_clarity = lambda_clarity
        self.use_protection = use_protection
        self.gate_quantile = gate_quantile
        self.support_frac = support_frac
        self.influence_mode = influence_mode
        self.use_head_protection = use_head_protection

    def train_task(
        self, model: MLP, params, causal_state: CausalState, task: Task, task_idx
    ):
        num_batches = task.train_X.shape[0] // self.batch_size

        # Determine already-seen classes for head protection
        seen_classes = tuple(causal_state.seen_classes) if self.use_head_protection else ()

        for epoch in range(self.epochs):
            total_loss = 0
            total_sparsity = 0

            for i in range(num_batches):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                batch_X = task.train_X[start:end]
                batch_y = task.train_y[start:end]

                if self.use_protection:
                    support_to_pass = causal_state.accumulated_support
                else:
                    support_to_pass = {}
                params, loss, sparsity = _gated_train_step(
                    params,
                    batch_X,
                    batch_y,
                    self.lr,
                    self.p,
                    self.kappa,
                    model,
                    self.lambda_clarity,
                    causal_state.all_gate_vectors,
                    support_to_pass,
                    self.influence_mode,
                    self.gate_quantile,
                    seen_classes,
                    self.use_head_protection,
                )
                total_loss += loss
                total_sparsity += sparsity

            print(
                f"Epoch: {epoch + 1} Loss: {total_loss/num_batches:.4f} | Sparsity: {total_sparsity/num_batches:.4f}"
            )

        pre_acts, _, _ = model.forward_with_states(params, batch_X)
        final_influence = estimate_influence(params, pre_acts,
                                             influence_mode=self.influence_mode)

        probe_X = task.train_X[: self.batch_size]

        gate_vecs = extract_gate_vectors(params, probe_X, model, self.p, self.kappa,
                                         gate_quantile=self.gate_quantile,
                                         influence_mode=self.influence_mode)

        new_support = compute_support_mask(gate_vecs, support_frac=self.support_frac)

        if self.use_protection:

            if len(causal_state.accumulated_support) == 0:
                new_accumulated_support = new_support
            else:
                new_accumulated_support = {}
                for layer_key in new_support:
                    new_accumulated_support[layer_key] = jnp.maximum(
                        causal_state.accumulated_support[layer_key],
                        new_support[layer_key],
                    )
        else:
            new_accumulated_support = {}

        # Update seen classes
        new_seen_classes = list(causal_state.seen_classes) + list(task.classes)

        new_all = causal_state.all_gate_vectors + [gate_vecs]
        causal_state = CausalState(
            old_params=params,
            influence_scores=final_influence,
            gate_vectors=gate_vecs,
            all_gate_vectors=new_all,
            accumulated_support=new_accumulated_support,
            seen_classes=new_seen_classes,
        )

        return params, causal_state, total_loss / num_batches
