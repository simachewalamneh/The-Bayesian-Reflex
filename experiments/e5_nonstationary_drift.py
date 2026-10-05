"""E5 - Nonstationarity  [NOT IMPLEMENTED]

Question : How does belief adapt to change, and can ONE knob trade tracking vs retention?
Protocol : E5-A scalar drift theta_{t+1}=theta_t+eta (Kalman filter oracle, sweep sigma_eta).
E5-B functional drift f_{t+1}=f_t+eta(x) with kernel-correlated eta; stream sin -> shifted-sin -> cos.
Forgetting factor lambda (VCL, paper 9.2): q_t ∝ p(y|w) q_{t-1}^lambda. Metrics: recovery steps, old-task retention.
Reference: Baselines: Kalman filter, change-point detector.
Pass/fail criteria must be fixed here BEFORE coding (see docs/experimental_plan.md).
"""


def main():
    raise NotImplementedError("e5_nonstationary_drift")


if __name__ == "__main__":
    main()
