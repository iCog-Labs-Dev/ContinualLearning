import jax.numpy as jnp

from core.metrics import bce_nll, class_il_predict, nll, task_il_predict


class Evaluator:

    def evaluate(self, model, params, task, allowed_classes=None):
        logits = model.forward(params, task.test_X)

        if allowed_classes is not None:
            predictions = task_il_predict(logits, allowed_classes)
        else:
            predictions = class_il_predict(logits)

        accuracy = float(jnp.mean(predictions == task.test_y))
        nll_value = float(nll(logits, task.test_y, allowed_classes))
        # Secondary Bernoulli BCE-NLL is Task-IL only; NaN under Class-IL.
        if allowed_classes is not None:
            bce_value = float(bce_nll(logits, task.test_y, allowed_classes))
        else:
            bce_value = float("nan")
        return accuracy, nll_value, bce_value

    def compute_baselines(self, model, params, tasks):
        # Untrained-model accuracy AND NLL baselines, for both protocols.
        # NLL baselines feed forward-transfer in NLL space.
        class_il = [self.evaluate(model, params, t) for t in tasks]
        task_il = [self.evaluate(model, params, t, t.classes) for t in tasks]
        class_il_acc = [acc for acc, _, _ in class_il]
        task_il_acc = [acc for acc, _, _ in task_il]
        class_il_nll = [nll_v for _, nll_v, _ in class_il]
        task_il_nll = [nll_v for _, nll_v, _ in task_il]
        return class_il_acc, task_il_acc, class_il_nll, task_il_nll
