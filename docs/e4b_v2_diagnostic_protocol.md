# E4b v2 DIAGNOSTIC protocol (declared before running; v1 results and verdict stay as recorded: 14/23)

Purpose: separate numerical-floor / boundary / out-of-regime effects from the substantive Markov-approximation limitation found in v1. These are NEW criteria with a stated technical reason each; they do not replace v1 and they do not change the v1 verdict. They are tested on new data where possible (a second truth function, new inputs, wider grids). If any v2 criterion fails it is reported as failed.

## V1 numerical integrity (explains v1 T1)
Reason: the residual of an explicit inverse scales like cond(A)*eps.
- V1a Cholesky solve residual < 1e-6 for N = 20, 100, 200 (authoritative path).
- V1b explicit-inverse residual <= 100 * cond(A) * 2.2e-16 (explained by conditioning; reported, not accepted as usable).

## V2 Gaussian-kernel error floor (explains v1 T2a Gaussian mean misses)
Reason: the jitter acts as a nugget; for a truth in the RKHS the bias scales like sqrt(jitter). Truths: eta*(y) and a NEW truth eta2(y)=0.5 sin(1.3y)+0.3 cos(0.7y)+0.1; n in {20,40,80,160,320}; jitter in {1e-8,1e-10,1e-12}; sup error on a 2001-point mesh.
- V2a floor bound: sup mean error floor (min over n>=40) at jitter 1e-8 <= 3*sqrt(1e-8) = 3e-4.
- V2b floor scales with the jitter: floor(1e-8)/floor(1e-10) >= 3 and floor(1e-10)/floor(1e-12) >= 3 (both truths).
- V2c plateau: for each jitter, max/min of the error over n>=40 is <= 3 (a reproducible floor, not a trend).

## V3 exponential-kernel asymptotics (explains v1 T2a exponential misses)
Reason: sup error is dominated by edge extrapolation (O(h)); the interior should show the interpolation order (O(h^2)). n in {20,...,1280}, mesh 20001 points, interior |z|<=2.5, boundary strip |z|>3-1.5h.
- V3a interior sup mean-error slope in [-2.3,-1.7]; V3b interior L2 slope in [-2.3,-1.7]; V3c boundary-strip sup slope in [-1.3,-0.7].
- V3d variance slope fitted on n>=80 is <= -0.95 (asymptotic regime of the 'at least O(1/n)' claim).

## V4 Markov approximation: mechanism, not an excuse (v1 T2b stays a finding)
- V4a mechanism: for two inputs at separation Delta in {1e-4,1e-3,1e-2,1e-1,1} and N in {20,40,80,160} (jitter 1e-12), KL(exact||LUT) vs r/Delta^2 (r = LUT residual variance at the second input): Spearman rho(log KL, log r/Delta^2) > 0.95 over pairs with KL>1e-12. This tests the hypothesis that the LUT fails when its independent residual noise exceeds the variation the exact model leaves between nearby inputs.
- V4b separated inputs (NEW set linspace(-2.4,2.9,6), jitter 1e-12): Gaussian KL non-increasing (tolerance 1e-11) and KL_80 < 1e-9; control KL_5 > 1e-2 and > 100*KL_80; exponential kernel KL <= 1e-10 for N>=20 (inputs in different cells, Markov kernel).
- V4c the clustered-trajectory KL is not jitter-dominated: KL_80(jitter 1e-12)/KL_80(jitter 1e-10) in [0.5,2].

## Conclusion rule
v2 passes only if every criterion above passes. Even then the v1 T2b result stands: for converging dynamics at these grids the Markov approximation is not arbitrarily accurate; V4 tests why, not whether.

## Results (run after declaring the criteria above)
19/21 pass. V1a/V1b pass (Cholesky residual 2.5e-8 / 9.6e-8 / 1.7e-7 for N=20/100/200; the explicit-inverse residuals 2.1e-7 / 1.4e-5 / 3.5e-5 are within 100*cond*eps). V2a/V2b pass for both truths: the Gaussian mean-error floor at jitter 1e-8 is 9.8e-6 (eta*) and 4.5e-6 (eta2) at n=320, and it falls by 10.7x and 10.8x per 100x reduction of the jitter, i.e. it scales like sqrt(jitter). V3a-d pass: exponential interior slopes -2.0 (sup) and -2.0 (L2), boundary strip -0.97, variance (n>=80) -0.988. V4a: Spearman 1.0 over 12 pairs (log KL vs log r/Delta^2); V4b: separated-input Gaussian KL_80 = 3.0e-13, control KL_5 = 0.036 (> 100 x KL_80), exponential KL <= 8.9e-16; V4c: clustered KL_80 changes by only 1.09x between jitter 1e-12 and 1e-10.
MISSES (reported, premise wrong): V2c plateau (max/min over n>=40: 7.5-27.4). The error does not plateau: at jitter 1e-8 it is 7.4e-5, 2.6e-5, 1.5e-5, 9.8e-6 for n=40, 80, 160, 320 (and 2.3e-6 -> 8.5e-8 at jitter 1e-12). So the v1 Gaussian miss is a jitter-induced regularisation bias (sqrt(jitter) scaling, slowly improving with n), not a machine-precision floor; the v1 criterion (<1e-6 at n=40, jitter 1e-8) was unattainable for that reason.
Caveats: V4a has only 12 points and both axes vary monotonically, so it shows association, not a quantitative law; one truth map, one kernel length-scale, 1-D.
Reading: the v1 numerical/boundary misses are explained and reproduced in their proper regimes. The Markov-approximation limitation is real and mechanistic (clustered, converging states), and it disappears for separated inputs.
