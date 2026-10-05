"""E8a - Ellipsoidal decomposition sampler, standalone (paper sec 3)  [NOT IMPLEMENTED]

Question : Does the sampler produce practically exact iid draws?
Protocol : Targets: N(0,I), Student-t(5), Cauchy, 2-component mixture; d=1..100. Track TV bound eps, minorisation p_i=s_i/S_i, coalescence time, runtime.
Reference: Reference = exact samplers (known closed forms).
Pass/fail criteria must be fixed here BEFORE coding (see docs/experimental_plan.md).
"""


def main():
    raise NotImplementedError("e8a_ellipsoidal_sampler")


if __name__ == "__main__":
    main()
