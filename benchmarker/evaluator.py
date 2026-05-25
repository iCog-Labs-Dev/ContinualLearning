import jax.numpy as jnp


class Evaluator:

    def evaluate(self, model, params, task, allowed_classes=None):
        logits = model.forward(params, task.test_X)

        if allowed_classes is not None:
            mask = jnp.full((logits.shape[1],), -jnp.inf)
            mask = mask.at[jnp.array(allowed_classes)].set(0.0)
            logits = logits + mask

        predictions = jnp.argmax(logits, axis=1)
        return float(jnp.mean(predictions == task.test_y))

    def compute_baselines(self, model, params, tasks):
        class_il_baselines = [self.evaluate(model, params, t) for t in tasks]
        task_il_baselines = [self.evaluate(model, params, t, t.classes) for t in tasks]
        return class_il_baselines, task_il_baselines

    def evaluate_all(self, model, params, tasks):
        class_il_row = [self.evaluate(model, params, t) for t in tasks]
        task_il_row = [self.evaluate(model, params, t, t.classes) for t in tasks]
        return class_il_row, task_il_row
