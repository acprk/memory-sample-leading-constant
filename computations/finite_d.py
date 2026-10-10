"""Finite-d evaluation of the improved proof (moving split, terminal C=1).
For given d, (kappa,pi,mu) and memory M(d), computes the exact per-block cost
  ln G = ln(w)/m + max_{1<=b<=k} [ (1/m) sum_j ln factor_j(b) + ln E rho^{-p}(b) ],
over test dimensions D in {d, d-2J}, and the resulting lower bound
  T >= k[(p lam ln2 - ln 1.5)/ln G - 1].
Also checks monotonicity of the block cost in b (the short last batch).
usage: python3 finite_d.py d kappa pi mu [Mexp]   (M = d^Mexp bits, default 1.5)"""
import sys
from math import log, log2, ceil
from costs import ln_block, T_lower, ln_residual

d = int(float(sys.argv[1])); kap, pi, mu = map(float, sys.argv[2:5])
Mexp = float(sys.argv[5]) if len(sys.argv) > 5 else 1.5
k = int(kap * d); l = int(pi * d / 2); m = max(1, int(mu * d)); p = 2 * l
M = d ** Mexp
lam_max = 50.0
LN2 = log(2)
# crude J bound used only for the dimension range: T < 2 ln2 d lam
Jmax = ceil(2 * LN2 * d * lam_max / k) + 1
Dset = sorted({d, d - 2 * Jmax})
assert min(Dset) - 1 >= p, "terminal needs p <= D-1"
bs = sorted(set([1, 2, 3, max(2, k // 4), max(2, k // 2), max(2, 3 * k // 4), k]))
jstep = max(1, m // 200)
worst = -1e300; prev = None; mono = True
for D in Dset:
    prev = None
    for b in bs:
        v = ln_block(D, b, m, l, "opt", jstep=jstep)
        if prev is not None and v < prev - 1e-9:
            mono = False
        prev = v
        worst = max(worst, v)
print(f"d={d} k={k} l={l} m={m} D in {Dset}  block cost monotone in b: {mono}")
for tag, Mbits in (("M=0", 0.0), (f"M=d^{Mexp}", M)):
    for lam in (log2(10), 10.0, lam_max):
        T = (lam * 2 * LN2 * d * 1.0)
        w = (T + 2) * 2 ** 0  # log2 w = M + log2(T+2)
        lnw = Mbits * LN2 + log(T + 2)
        lnG = lnw / m + worst
        TLB = T_lower(d, k, l, lnG, lam, 0.0)
        print(f"  {tag:10s} lam={lam:7.3f}: lnG/d={lnG/d:.6e}  T_LB/(d lam)={TLB/(d*lam):.5f}   (2ln2={2*LN2:.5f})")
