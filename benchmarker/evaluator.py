import jax
import jax.numpy as jnp


class Evaluator:

    def evaluate(self, model, params, task, allowed_classes=None):
        logits = model.forward(params, task.test_X)

        if allowed_classes is not None:
            mask = jnp.full((logits.shape[1],), -jnp.inf)
            mask = mask.at[jnp.array(allowed_classes)].set(0.0)
            logits = logits + mask

        predictions = jnp.argmax(logits, axis=1)
        accuracy = float(jnp.mean(predictions == task.test_y))

        log_probs = jax.nn.log_softmax(logits, axis=1)
        true_log_probs = log_probs[jnp.arange(task.test_y.shape[0]), task.test_y]
        nll = float(-jnp.mean(true_log_probs))

        return accuracy, nll

    def compute_baselines(self, model, params, tasks):
        class_il_baselines = [self.evaluate(model, params, t)[0] for t in tasks]
        task_il_baselines = [self.evaluate(model, params, t, t.classes)[0] for t in tasks]
        return class_il_baselines, task_il_baselines

    def evaluate_all(self, model, params, tasks):
        class_il_acc, task_il_acc = [], []
        class_il_nll, task_il_nll = [], []
        for t in tasks:
            acc_c, nll_c = self.evaluate(model, params, t)
            acc_t, nll_t = self.evaluate(model, params, t, t.classes)
            class_il_acc.append(acc_c)
            task_il_acc.append(acc_t)
            class_il_nll.append(nll_c)
            task_il_nll.append(nll_t)
        return class_il_acc, task_il_acc, class_il_nll, task_il_nll
