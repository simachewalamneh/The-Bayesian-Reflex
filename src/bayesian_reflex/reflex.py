"""Orchestrator: Observe -> Predict -> Error -> Update -> Act (paper Fig. 1)."""


class BayesianReflex:
    def __init__(self, model, pc_layer, policy=None):
        self.model, self.pc, self.policy = model, pc_layer, policy

    def step(self, y, context=None):
        y_hat, var_pred = self.model.predict(context)      # prediction BEFORE seeing the update
        z = self.pc.standardized_error(y, y_hat, var_pred)
        self.model.update(y, context)
        action = self.policy(self.model) if self.policy else None   # E6+: uncertainty-driven
        return dict(y_hat=y_hat, var_pred=var_pred, z=z, action=action)
