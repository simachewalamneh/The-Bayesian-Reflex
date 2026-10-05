"""E6 - Uncertainty-driven action  [NOT IMPLEMENTED]

Question : Does uncertainty guide sampling/decisions better than passive baselines?
Protocol : E6-A uncertainty sampling x=argmax Var[f(x)|D]; E6-B information gain (differs from A only for batch/heteroscedastic/integrated-variance);
E6-C Thompson sampling, regret on Bernoulli/Beta then GP (paper 10.2). Optional E6-D: derivative-process design (paper 11). Baseline: random.
Reference: Metrics: RMSE vs queries, cumulative regret.
Pass/fail criteria must be fixed here BEFORE coding (see docs/experimental_plan.md).
"""


def main():
    raise NotImplementedError("e6_uncertainty_action")


if __name__ == "__main__":
    main()
