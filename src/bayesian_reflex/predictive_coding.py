"""Predictive-coding layer: REPORTS the discrepancy between belief-generated prediction and
observation. It does not perform the Bayes update (design decision pending, see docs/experimental_plan.md).
"""
import numpy as np


class PredictiveCodingLayer:
    @staticmethod
    def prediction_error(y, y_hat):
        return y - y_hat

    @staticmethod
    def predictive_precision(var_pred):
        return 1.0 / var_pred

    def precision_weighted_error(self, y, y_hat, var_pred):
        return self.predictive_precision(var_pred) * self.prediction_error(y, y_hat)

    def standardized_error(self, y, y_hat, var_pred):
        """z_t = (y - y_hat) / sqrt(var_pred); var_pred = sigma_{t-1}^2 + sigma_y^2 (one-step-ahead)."""
        return self.prediction_error(y, y_hat) / np.sqrt(var_pred)
