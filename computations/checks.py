"""Numerical checks of the lemmas in paper/main.tex.
(a) Monte Carlo: Q-law of the actual next direction (Beta(alpha,beta0)) and of nu1 (Beta(alpha,beta1)).
(b) Pointwise density domination with the moving split q0 (grid, exact Beta densities).
(c) Terminal cap: sigma{angle(u,w)<=arcsin delta} <= delta^p for all p<=D-1, delta<1 (grid).
(d) Kaczmarz (real registers): T needed for angular error 0.1 at prob 2/3 vs d ln(1/eps^2).
(e) E Z^{-l} product formula vs Monte Carlo.
usage: python3 checks.py   (single-threaded, ~1-2 min)"""
import numpy as np
from scipy import stats, special
from math import log, exp, asin, lgamma
from costs import ln_factor, mix_terms, lB
rng = np.random.default_rng(20261009)

def unif_sphere(n, size):
    x = rng.standard_normal((size, n)); return x / np.linalg.norm(x, axis=1, keepdims=True)

# (a)
Dp, n, j = 40, 30, 5          # dim E, dim N, dim V
alpha, a_s, b0, b1 = (n - j) / 2, (Dp - j) / 2, j / 2, (Dp - n + j - 1) / 2
S = 20000; Qv = np.empty(S); Q1 = np.empty(S)
V = np.eye(Dp)[:, :j]
for s in range(S):
    # N = V + uniform (n-j)-subspace of V^perp ; v uniform on S(N)
    G = rng.standard_normal((Dp - j, n - j)); Qo, _ = np.linalg.qr(G)
    coef = rng.standard_normal(n); coef /= np.linalg.norm(coef)
    Qv[s] = np.sum(coef[j:] ** 2)          # |P_{V^perp} v|^2
    # nu1: sqrt(1-Z) e0 + sqrt(Z) u0, e0 in S(V), u0 uniform on S(E cap e0^perp)
    Z = rng.beta(alpha, a_s - alpha)
    e0 = np.zeros(Dp); e0[:j] = unif_sphere(j, 1)[0]
    u = rng.standard_normal(Dp); u -= u.dot(e0) * e0; u /= np.linalg.norm(u)
    y = np.sqrt(1 - Z) * e0 + np.sqrt(Z) * u
    Q1[s] = np.sum(y[j:] ** 2)
print("(a) KS actual Q vs Beta(alpha,beta0): p =", stats.kstest(Qv, stats.beta(alpha, b0).cdf).pvalue)
print("(a) KS nu1   Q vs Beta(alpha,beta1): p =", stats.kstest(Q1, stats.beta(alpha, b1).cdf).pvalue)

# (b)
worst = -np.inf
for (D, b, j, l) in [(400, 40, 3, 10), (400, 40, 30, 10), (2000, 50, 10, 100), (2000, 200, 1, 50), (5000, 100, 20, 500)]:
    eA, eB, lA0, lB0, lEZ = mix_terms(D, b, j, l)
    Dp = D - 1; nn = D - b
    a, a_s, b0, b1 = (nn - j) / 2, (Dp - j) / 2, j / 2, (b + j - 2) / 2
    # recover optimal q0 by scanning
    us = np.exp(np.linspace(-20, 3, 4000)); best = None
    for u in us:
        q0 = -np.expm1(-u)
        lA = lA0 - eA * np.log(q0); lBB = lB0 + eB * u
        v = np.logaddexp(lA, lBB + lEZ)
        if best is None or v < best[0]: best = (v, q0, lA, lBB)
    v, q0, lA, lBB = best
    assert ln_factor(D, b, j, l, "opt") <= v + 1e-9, (v, ln_factor(D, b, j, l, "opt"))  # golden search at least as good as scan
    q = np.concatenate([np.linspace(1e-6, 1 - 1e-12, 200001), 1 - np.logspace(-14, -1, 20000)])
    lf = lambda A, B: (A - 1) * np.log(q) + (B - 1) * np.log1p(-q) - lB(A, B)
    lhs = lf(a, b0)
    rhs = np.where(q >= q0, lA + lf(a_s, b0), lBB + lf(a, b1))
    worst = max(worst, np.max(lhs - rhs))
print("(b) max_q [ln f_v - ln(dominating density)] over cases =", worst, "(must be <= ~0)")

# (c)
bad = 0
for D in [3, 5, 10, 50, 200, 1000]:
    for de in np.linspace(0.001, 0.999, 999):
        th = asin(de)
        mass = 0.5 * special.betainc((D - 1) / 2, 0.5, np.sin(th) ** 2)   # cap of angular radius th<pi/2
        for p in set([1, max(1, (D - 1) // 2), D - 1]):
            if mass > de ** p * (1 + 1e-12): bad += 1
print("(c) terminal cap violations of  mass <= delta^p :", bad)

# (d)
def kacz_success(d, T, eps, trials=300):
    ok = 0
    for _ in range(trials):
        s = unif_sphere(d, 1)[0]; z = np.zeros(d)
        X = rng.standard_normal((T, d)); y = X @ s
        for t in range(T):
            x = X[t]; z += (y[t] - x @ z) / (x @ x) * x
        nz = np.linalg.norm(z)
        ok += (nz > 0 and np.arccos(np.clip(z @ s / nz, -1, 1)) <= eps)
    return ok / trials
d, eps = 100, 0.1
T0 = d * log(1 / eps ** 2)
for f in (0.8, 0.9, 1.0, 1.1, 1.2):
    print(f"(d) Kaczmarz d={d} eps={eps}: T={int(f*T0)} ({f:.1f} x d ln(1/eps^2)) success={kacz_success(d, int(f*T0), eps):.3f}")

# (e)
A, Atot, l = 30.0, 33.5, 7
Zs = rng.beta(A, Atot - A, 2_000_000)
print("(e) E Z^{-l}: MC", np.mean(Zs ** (-l)), " formula", exp(lgamma(Atot) + lgamma(A - l) - lgamma(A) - lgamma(Atot - l)))
