"""E3 - Precision-weighted prediction error  [NOT IMPLEMENTED]

Question : Is z_t = (y_t - yhat_t)/sqrt(sigma_{t-1}^2 + sigma_y^2) ~ N(0,1) and uncorrelated over t under the correct model?
Protocol : Mean~0, var~1, KS/QQ pass, autocorrelation~0. Kalman identity: update = gain * error, gain = s^2/(s^2+sy^2).
Negative control: wrong sigma_y must break N(0,1). Unknown-variance variant gives Student-t z (paper 5.2).
Reference: Compare raw error e_t vs precision-weighted z_t at sy=0.1 vs 1.0.
Pass/fail criteria must be fixed here BEFORE coding (see docs/experimental_plan.md).
"""


def main():
    raise NotImplementedError("e3_prediction_error")


if __name__ == "__main__":
    main()
