"""Effective constant when the memory term is kept: M = A d^2 bits adds A ln2/mu (nats per d) per block.
c(A) = sup_{kappa,pi,mu} kappa*pi*ln2/(g_opt(kappa,pi,mu) + A ln2/mu)   (lambda->inf, d->inf, J=o(d) assumed).
Shows how slowly the o(d^2) hypothesis converges.  usage: python3 memratio.py"""
import numpy as np
from math import log
from scipy.optimize import minimize
from rates import g
LN2 = log(2)
for A in (1e-2, 1e-3, 1e-4, 1e-6, 1e-8):
    f = lambda y: (1e9 if np.exp(y).min() < 1e-9 else -(lambda k, p, m: k*p*LN2/(g(k, p, m, "opt") + A*LN2/m))(*np.exp(y)))
    best = None
    for x0 in [(0.1, 0.3, 0.3), (0.03, 0.2, 0.1), (0.01, 0.1, 0.05)]:
        r = minimize(f, np.log(x0), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-12, "maxiter": 3000})
        if best is None or r.fun < best.fun: best = r
    print(f"M = {A:g} d^2 : c(A) ~ {-best.fun:.5f}  at (kappa,pi,mu) = {np.exp(best.x)}")
