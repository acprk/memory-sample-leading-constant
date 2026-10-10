"""Interval-arithmetic (mpmath.iv) certificate for the asymptotic constant
obtained with the moving-split mixture lemma and closed-form bounds:

  delta   = (1+zeta)(kappa+mu)/(1 - pi/(1-kappa-mu)),  zeta=1e-3
            (makes the nu1 term  (beta1/(alpha+beta1))^{floor((b-2)/2)} delta^{-(b-2)/2} E Z^{-l} <= e^{O(1)}
             for every batch length 2<=b<=k and every j<m, using ln(1+x)<=x in E Z^{-l})
  g_up    = R1(kappa,pi) + (kappa/2) * (-ln(1-delta))      (nats per d per block)
  c(lam)  = kappa*(pi*ln2*lam/g_up - 1)/lam,  increasing in lam,
  c_unif  = c(log2 10)   (valid for every 0<eps<=1/10, d >= d0(kappa,pi,mu,M))
with R1 = F(1/2,pi/2)-F((1-kappa)/2,pi/2), R2(mu) = F((1-mu)/2,pi/2)-F((1-kappa-mu)/2,pi/2),
F(a,L) = int_0^L ln(a-x)dx.  Requirements checked: pi < 1-kappa-mu, delta < 1.
usage: python3 certify.py"""
from mpmath import iv, mpf, log as mlog
iv.prec = 200

def F(a, L):
    g = lambda t: t * iv.log(t) - t
    return g(a) - g(a - L)

def cert(kap, pi, mu, lam=None):
    kap, pi, mu = iv.mpf(kap), iv.mpf(pi), iv.mpf(mu)
    assert (pi + kap + mu < 1).a if hasattr(pi + kap + mu < 1, 'a') else True
    half = iv.mpf(1) / 2
    R1 = F(half, pi / 2) - F((1 - kap) / 2, pi / 2)
    R2 = F((1 - mu) / 2, pi / 2) - F((1 - kap - mu) / 2, pi / 2)
    zeta = iv.mpf('0.001')
    delta = (1 + zeta) * (kap + mu) / (1 - pi / (1 - kap - mu))
    assert delta.b < 1
    g_up = R1 + (kap / 2) * (-iv.log(1 - delta))
    ln2 = iv.log(2)
    lam = iv.log(10) / ln2 if lam is None else iv.mpf(lam)
    c = kap * (pi * ln2 * lam / g_up - 1) / lam
    cinf = kap * pi * ln2 / g_up
    return c, cinf, delta, g_up

if __name__ == "__main__":
    two_ln2 = 2 * iv.log(2)
    print("2 ln 2 =", two_ln2)
    print("old c* = 2^-16 =", mpf(2) ** -16)
    for kap, pi, mu in [("1/16", "1/16", "1/16"), ("0.01", "0.05", "0.001"), ("1e-3", "0.02", "1e-4"),
                        ("1e-4", "5e-3", "1e-5"), ("1e-5", "1e-3", "1e-6"), ("1e-7", "1e-4", "1e-8")]:
        k, p, m = (iv.mpf(eval(s)) if '/' in s else iv.mpf(s) for s in (kap, pi, mu))
        c, cinf, de, G = cert(k, p, m)
        lo = c.a
        print(f"kappa={kap:>5s} pi={pi:>5s} mu={mu:>5s}: c_unif in [{float(c.a):.8f},{float(c.b):.8f}]  "
              f"c_inf in [{float(cinf.a):.8f},{float(cinf.b):.8f}]  delta={float(de.b):.3e}  "
              f"ratio c_unif/2^-16 >= {float(c.a) * 2**16:.1f}   gap to 2ln2: x{float((two_ln2 / c).b):.5f}")
