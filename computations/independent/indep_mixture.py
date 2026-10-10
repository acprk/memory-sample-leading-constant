"""Independent checks of the mixture lemma, re-implemented from scratch (does not import ../costs.py).
1) vector-level MC: Q-law of actual next direction vs Beta(alpha,beta0); nu1 law vs Beta(alpha,beta1)
2) pointwise density domination  f_{a,b0} <= A f_{a*,b0} + B f_{a,b1}  with q0 = 1-delta, small & large dims,
   including extreme q0 -> 1 and b odd/even, j=1, j=m-1
3) exact (mpmath) per-direction factor A + B*E Z^{-l} at large d vs claimed (1-delta)^{-k/2}+e^{O(1)}
"""
import numpy as np, mpmath as mp
from scipy import stats
rng = np.random.default_rng(777)
mp.mp.dps = 40

def lbeta(a, b): return mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)

# ---------- 1) MC -----------
def mc(Dp, n, j, S=40000):
    a, a_s, b0, b1 = (n-j)/2, (Dp-j)/2, j/2, (Dp-n+j-1)/2
    # actual: V=span(e_1..e_j); N = V + random (n-j)-subspace W of V^perp (dim Dp-j); v unif on S(N)
    Qv = np.empty(S)
    for s in range(S):
        W, _ = np.linalg.qr(rng.standard_normal((Dp-j, n-j)))
        g = rng.standard_normal(n); g /= np.linalg.norm(g)
        w = W @ g[j:]                      # component in V^perp (coords j..Dp-1)
        Qv[s] = w @ w
    # nu1
    Z = rng.beta(a, a_s - a, S)
    e0 = rng.standard_normal((S, j)); e0 /= np.linalg.norm(e0, axis=1, keepdims=True)
    u = rng.standard_normal((S, Dp)); E0 = np.zeros((S, Dp)); E0[:, :j] = e0
    u -= (u*E0).sum(1, keepdims=True)*E0; u /= np.linalg.norm(u, axis=1, keepdims=True)
    y = np.sqrt(1-Z)[:, None]*E0 + np.sqrt(Z)[:, None]*u
    Q1 = (y[:, j:]**2).sum(1)
    return (stats.kstest(Qv, stats.beta(a, b0).cdf).pvalue,
            stats.kstest(Q1, stats.beta(a, b1).cdf).pvalue if b1 > 0 else None)

print("== 1) MC Q-laws (KS p-values; want not tiny)")
for (Dp, n, j) in [(12, 9, 1), (12, 9, 4), (30, 20, 7), (25, 23, 3), (8, 6, 2)]:
    print((Dp, n, j), mc(Dp, n, j))

# ---------- 2) pointwise domination ----------
def dom_check(Dp, n, j, delta, ngrid=20001):
    a, a_s, b0, b1 = mp.mpf(n-j)/2, mp.mpf(Dp-j)/2, mp.mpf(j)/2, mp.mpf(Dp-n+j-1)/2
    b = Dp + 1 - n
    q0 = 1 - mp.mpf(delta)
    A = mp.exp(lbeta(a_s, b0) - lbeta(a, b0)) * q0**(-mp.mpf(b-1)/2)
    B = mp.exp(lbeta(a, b1) - lbeta(a, b0)) * (1-q0)**(-mp.mpf(b-2)/2)
    worst = -mp.inf
    qs = [mp.mpf(i)/ngrid for i in range(1, ngrid)] + [q0, q0 - mp.mpf(10)**-12, q0 + mp.mpf(10)**-12]
    for q in qs:
        lf = (a-1)*mp.log(q) + (b0-1)*mp.log(1-q) - lbeta(a, b0)
        lf0 = (a_s-1)*mp.log(q) + (b0-1)*mp.log(1-q) - lbeta(a_s, b0)
        lf1 = (a-1)*mp.log(q) + (b1-1)*mp.log(1-q) - lbeta(a, b1)
        rhs = mp.log(A*mp.exp(lf0) + B*mp.exp(lf1))
        worst = max(worst, lf - rhs)
    return float(worst), float(A), float(B)

print("== 2) max_q ln(f/(A f0 + B f1)) (must be <= 0)")
for (Dp, n, j, de) in [(12, 9, 1, .3), (12, 9, 4, .01), (40, 30, 5, 1e-3), (40, 37, 1, 1e-6), (40, 38, 39-38+1, .5),
                       (200, 150, 20, 1e-4), (200, 199, 5, 1e-9), (300, 260, 100, 0.2)]:
    if j >= n: continue
    print((Dp, n, j, de), dom_check(Dp, n, j, de, ngrid=4001))

# ---------- 3) exact per-direction factor at large d ----------
def per_dir(D, b, j, l, delta):
    Dp, n = D-1, D-b
    a, a_s, b0, b1 = mp.mpf(n-j)/2, mp.mpf(Dp-j)/2, mp.mpf(j)/2, mp.mpf(Dp-n+j-1)/2
    lA = lbeta(a_s, b0) - lbeta(a, b0) - mp.mpf(b-1)/2*mp.log(1-delta)
    lB = lbeta(a, b1) - lbeta(a, b0) - mp.mpf(b-2)/2*mp.log(delta)
    lEZ = mp.loggamma(a-l) + mp.loggamma(a_s) - mp.loggamma(a) - mp.loggamma(a_s-l)
    return lA, lB + lEZ

print("== 3) exact per-direction ln(A), ln(B E Z^-l) at large d; claimed ln A <= (k/2)(-ln(1-delta)), ln(B EZ) = O(1)")
for (d, kap, pi, mu) in [(10**6, 1e-3, 2e-2, 1e-4), (10**7, 1e-4, 5e-3, 1e-5), (10**8, 1e-5, 1e-3, 1e-6)]:
    k, m, l = int(kap*d), int(np.ceil(mu*d)), int(pi*d/2)
    zeta = 1e-3
    delta = mp.mpf((1+zeta)*(kap+mu)/(1-pi/(1-kap-mu)))
    worstB = -mp.inf; worstA = -mp.inf; arg = None
    for b in sorted(set([2, 3, 4, 5, k//2, k-1, k])):
        for j in sorted(set([1, 2, m//2, m-1])):
            for D in [d, d - 2*int(10*np.log(d))]:
                lA, lBZ = per_dir(D, b, j, l, delta)
                worstA = max(worstA, lA)
                if lBZ > worstB: worstB, arg = lBZ, (b, j, D)
    print(f"d={d} kap={kap} pi={pi} mu={mu}: max lnA={float(worstA):.4f} vs (k/2)(-ln(1-delta))={float(-k/2*mp.log(1-delta)):.4f};"
          f" max ln(B EZ^-l)={float(worstB):.4f} at {arg}")
    # what if zeta=0 or delta smaller (falsify necessity)
    for zz in [0.0, -0.05]:
        dl = mp.mpf((1+zz)*(kap+mu)/(1-pi/(1-kap-mu)))
        wb = max(per_dir(d, k, m-1, l, dl)[1], per_dir(d, k, 1, l, dl)[1])
        print(f"    zeta={zz}: ln(B EZ) at b=k = {float(wb):.3f}")
