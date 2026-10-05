"""E2 - Sequential = batch (exact oracle)  [NOT IMPLEMENTED]

Question : Does the recursive update recover the exact batch posterior (paper Thm 2.1) with no information loss?
Protocol : Conjugate scalar: |mu_seq-mu_batch|, |var_seq-var_batch| ~ 1e-12 (machine precision), KL ~ 0 for T up to 100.
GP (with E4): sequential rank-one Cholesky update vs batch GP, tolerance ~1e-6.
Reference: Oracle test, no learned model. Reference = analytic batch posterior.
Pass/fail criteria must be fixed here BEFORE coding (see docs/experimental_plan.md).
"""


def main():
    raise NotImplementedError("e2_sequential_equivalence")


if __name__ == "__main__":
    main()
