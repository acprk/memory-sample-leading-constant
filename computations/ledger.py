"""Loss ledger for c_* = 2^{-16} (07-beta-mixture.tex, proof of Theorem s:main).
Positive control: paper parameters k=m=floor(d/16), l=floor(d/32), C_term=8.
Prints effective constant c = T_LB/(d*lambda) after each successive correction.
The w^{1/m} memory term is o(d) for M=o(d^2) and is set to 0 (limit d->inf);
D is taken = d (the test dimension is d - 2J = d - o(d)).
usage: python3 ledger.py [d]"""
import sys
from math import log, log2
from costs import ln_block, T_lower

d = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
k = m = d // 16; l = d // 32; p = 2 * l
LN2 = log(2)

print(f"d={d}  k=m={k}  l={l}")
print("S0 paper literal: c=2^-16 =", 2**-16)
# S1: paper's own displayed bounds, honest arithmetic: log2 G = 4d (2^{3d}*2^d), k=p=d/16, C=8, lambda->inf
c1 = (k / d) * (p / d) / 4
print(f"S1 same bounds, no slack (k,p=d/16 not d/32; log2G=4d not 5d; lam-3~lam; lam->inf): c={c1:.3e} = 2^{log2(c1):.2f}")
rows = []
for name, mode in [("S2 exact value of paper-lemma bounds 2^{D'}(1+EZ^-l), worst j", "paper_worst"),
                   ("S3 proof constant 2^{(b-1)/2}(1+EZ^-l), all j", "half"),
                   ("S3b split at 1/2 with exact Beta ratios", "half_ex"),
                   ("S4 moving split q0 (optimised per j)", "opt")]:
    lnG = ln_block(d, k, m, l, mode, jstep=max(1, m // 400))
    for lnC, tag in [(log(8), "C=8"), (0.0, "C=1")]:
        for lam in (log2(10), 10.0, 1e3, 1e9):
            T = T_lower(d, k, l, lnG, lam, lnC)
            rows.append((name, tag, lam, lnG / d / LN2, T / (d * lam)))
for r in rows:
    print(f"{r[0]:62s} {r[1]}  lam={r[2]:10.4g}  log2G/d={r[3]:.5f}  c={r[4]: .4e}")
