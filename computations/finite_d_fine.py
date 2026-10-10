"""Finite-d effective constant c_eff = T_LB/(d lambda) of the improved proof, finer parameter grid.
Reuses the routine Tmin() from independent/indep_constant.py (exact beta products,
worst test dimension D = d - 2J, moving split delta=(1+zeta)(kappa+mu)/(1-pi/(1-kappa-mu)), terminal C=1,
memory term ln((T+2)2^M)/m).  Memory = Kaczmarz memory d(lambda + 2 log2 d).
Caveat (inherited): the block maximum is evaluated on b in {k-1,k}, j in {1,m/2,m-1}; this is an
estimate of the finite-d constant, not a certificate.
usage: python3 finite_d_fine.py"""
import sys, os, itertools
import numpy as np
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "independent", "indep_constant.py")).read()
_src = _src.split('if __name__ != "__main__": raise SystemExit')[0]
_src = _src.split('lam10 = mp.log(10)/LN2')[0] + 'def lbeta' + _src.split('def lbeta')[1]
class _NS: pass
ic = _NS(); _g = {}
exec(compile(_src, "indep_constant.py", "exec"), _g)
ic.Tmin = _g["Tmin"]

lam = 10
for d in [10**4, 10**6, 10**8, 10**10]:
    M = int(d * (lam + 2 * np.log2(d)))
    best = (0.0, None)
    kaps = np.geomspace(1e-5, 0.3, 16); pis = np.geomspace(5e-3, 0.7, 14); mus = np.geomspace(1e-6, 0.3, 16)
    for kap, pi, mu in itertools.product(kaps, pis, mus):
        if pi + kap + mu >= 0.95 or 1.001 * (kap + mu) / (1 - pi / (1 - kap - mu)) >= 0.99:
            continue
        T = ic.Tmin(d, lam, M, float(kap), float(pi), float(mu))
        if T is not None and T / (d * lam) > best[0]:
            best = (T / (d * lam), (float(kap), float(pi), float(mu)))
    print(f"d={d:.0e} lam={lam} M=d(lam+2log2 d): c_eff={best[0]:.4f} at (kappa,pi,mu)={best[1]}", flush=True)
