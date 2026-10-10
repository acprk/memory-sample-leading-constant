"""d -> infinity rate functions (nats per unit d) for the beta-mixture proof,
with k=kappa d, l = pi d/2 (p = pi d), m = mu d, D = d (J = o(d)), M = o(d^2).

  c_inf(kappa,pi,mu)   = kappa*pi*ln2 / g                     (lambda -> inf)
  c(lambda)            = kappa*(pi*ln2*lambda/g - 1)/lambda   (terminal constant C=1)
  g = R1 + (1/mu) int_0^mu phi(eta) d eta
mode 'opt' : moving split  phi = min_delta max(a,b)
mode 'half': paper proof constant  phi = (kappa/2)ln2 + max(0,R2)
mode 'paper': lemma statement phi = ln2 + max(0,R2)
usage: python3 rates.py"""
from math import log, log2, inf
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.integrate import quad

LN2 = log(2)

def H(x, y):  # lim (1/d) ln B(x d, y d)
    f = lambda t: t * log(t) if t > 0 else 0.0
    return f(x) + f(y) - f(x + y)

def F(a, L):  # int_0^L ln(a - x) dx
    g = lambda t: t * log(t) - t if t > 0 else 0.0
    return g(a) - g(a - L)

def _Rint(a, h, L):  # int_0^L ln((a-x)/(a-h-x)) dx, cancellation-free
    return quad(lambda x: -np.log1p(-h / (a - x)), 0, L, epsabs=0, epsrel=1e-12, limit=200)[0]

def R1(kap, pi):  # residual: int_0^{pi/2} ln((1/2-x)/((1-kap)/2-x)) dx
    return _Rint(0.5, kap / 2, pi / 2)

def R2(kap, pi, eta):  # E Z^{-l}:  int_0^{pi/2} ln((a*-x)/(a-x))
    return _Rint((1 - eta) / 2, kap / 2, pi / 2)

def df(x, h):  # f(x+h)-f(x), f(t)=t ln t, stable for small h
    return h * np.log(x + h) + (x * np.log1p(h / x) if x > 0 else 0.0)

def phi_opt(kap, pi, eta):
    if eta <= 0:
        return 0.0
    a_, as_, b0, b1 = (1 - kap - eta) / 2, (1 - eta) / 2, eta / 2, (kap + eta) / 2
    h = kap / 2
    cA = df(a_, h) - df(a_ + b0, h)              # H(a*,b0)-H(a,b0)
    cB = df(b0, h) - df(a_ + b0, h) + R2(kap, pi, eta)   # H(a,b1)-H(a,b0)+R2
    def obj(lu):  # delta = 1 - q0 = exp(lu)
        de = np.exp(lu)
        a = cA - (kap / 2) * np.log1p(-de)
        b = cB - (kap / 2) * lu
        return max(a, b)
    r = minimize_scalar(obj, bounds=(-40, -1e-9), method="bounded", options={"xatol": 1e-12})
    return min(r.fun, obj(log(0.5)))

def phi_half(kap, pi, eta):
    return 0.0 if eta <= 0 else (kap / 2) * LN2 + max(0.0, R2(kap, pi, eta))

def phi_paper(kap, pi, eta):
    return 0.0 if eta <= 0 else LN2 + max(0.0, R2(kap, pi, eta))

PHI = {"opt": phi_opt, "half": phi_half, "paper": phi_paper}

def g(kap, pi, mu, mode="opt"):
    if not (0 < kap and 0 < pi and pi < 1 - kap - mu):
        return inf
    ph = PHI[mode]
    x, w = np.polynomial.legendre.leggauss(16)
    mix = sum(wi * ph(kap, pi, mu * (xi + 1) / 2) for xi, wi in zip(x, w)) / 2
    return R1(kap, pi) + mix

def c_lam(x, lam, mode):
    kap, pi, mu = x
    G = g(kap, pi, mu, mode)
    if G == inf:
        return -inf
    return kap * (pi * LN2 * lam / G - 1) / lam

def best(lam, mode, x0):
    f = lambda y: (1e9 if np.exp(y).min() < 1e-8 else -c_lam(np.exp(y), lam, mode))
    r = minimize(f, np.log(x0), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-10, "maxiter": 1500})
    return -r.fun, np.exp(r.x)

if __name__ == "__main__":
    lam0 = log2(10)
    print("2ln2 =", 2 * LN2)
    print("paper params (1/16,1/16,1/16), lam->inf:")
    for mode in ("paper", "half", "opt"):
        print(f"  {mode:5s} c_inf = {c_lam((1/16,1/16,1/16), 1e12, mode):.6f}   c(lam0) = {c_lam((1/16,1/16,1/16), lam0, mode):.6f}")
    print("residual-only bound kappa*pi*ln2/R1 at (1/16,1/16):", (1/16)**2 * LN2 / R1(1/16, 1/16))
    for mode, x0 in (("paper", (0.3, 0.3, 0.3)), ("half", (0.2, 0.5, 0.01)), ("opt", (0.01, 0.05, 0.001))):
        for lam in (lam0, 10.0, 1e12):
            c, x = best(lam, mode, x0)
            print(f"framework optimum mode={mode:5s} lam={lam:8.4g}: c={c:.6f}  (kappa,pi,mu)={x}")
    print("\nmoving split, hierarchy mu<<kappa<<pi -> 2ln2 (lam=log2 10, i.e. uniform constant):")
    for kap, pi, mu in [(1e-2, 5e-2, 1e-3), (1e-3, 2e-2, 1e-4), (1e-4, 5e-3, 1e-5), (1e-5, 1e-3, 1e-6), (1e-6, 2e-4, 1e-7)]:
        print(f"  kappa={kap:g} pi={pi:g} mu={mu:g}: c_inf={c_lam((kap,pi,mu),1e12,'opt'):.6f}  c(lam0)={c_lam((kap,pi,mu),lam0,'opt'):.6f}  "
              f"[2ln2*pi/(-ln(1-pi))={2*LN2*pi/(-np.log1p(-pi)):.6f}]")
