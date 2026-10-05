"""E4 - Function-space belief: exact GP (oracle) vs PC-FSVI  [NOT IMPLEMENTED]

Question : Does PC-FSVI match the exact sequential GP in predictive mean AND uncertainty?
Protocol : Metrics: RMSE(mu), RMSE(sigma), predictive log-lik, interval coverage, latency/memory per update.
Target f(x)=sin(x). Reuse ../pc-fsvi-continual-learning (gp.py, variational.py).
Reference: Reference = exact GP.
Pass/fail criteria must be fixed here BEFORE coding (see docs/experimental_plan.md).
"""


def main():
    raise NotImplementedError("e4_function_space")


if __name__ == "__main__":
    main()
