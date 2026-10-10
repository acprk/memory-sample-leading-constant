"""Independent re-implementation: (f) terminal arcsin-cap lemma; Kaczmarz first-order constant (real and finite-bit)."""
import numpy as np, mpmath as mp
from scipy import special
rng = np.random.default_rng(4242)
# (f1) cap of angular radius arcsin(rho) on S^{D-1}: mass = 1/2 I_{rho^2}((D-1)/2,1/2); claim <= rho^{D-1} (hence <= rho^p, p<=D-1)
worst = -1e9; arg = None
for D in range(2, 400):
    for rho in np.concatenate([np.linspace(1e-4, 1, 2000), 1 - np.logspace(-12, -1, 200)]):
        mass = 0.5 * special.betainc((D - 1) / 2, 0.5, rho**2)
        r = np.log(mass) - (D - 1) * np.log(rho) if mass > 0 else -np.inf
        if r > worst: worst, arg = r, (D, rho)
print("(f1) max ln(cap/rho^{D-1}) =", worst, "at", arg, "(<= -ln2 expected)")
# (f2) sup over centers w of sigma{u: |u-w|<=rho} equals arcsin-cap: MC in D=3,5 with centers at distance t
for D in [3, 5, 8]:
    u = rng.standard_normal((400000, D)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    for rho in [0.3, 0.7, 0.95]:
        best = 0; bt = None
        for t in np.linspace(0.05, 2.0, 60):
            m = np.mean(np.linalg.norm(u - np.eye(D)[0] * t, axis=1) <= rho)
            if m > best: best, bt = m, t
        print(f"(f2) D={D} rho={rho}: sup_t MC={best:.5f} (t*={bt:.3f}, sqrt(1-rho^2)={np.sqrt(1-rho**2):.3f}) "
              f"arcsin-cap={0.5*special.betainc((D-1)/2,.5,rho**2):.5f}  rho^(D-1)={rho**(D-1):.5f}")
# Kaczmarz: exact per-step E ln(1-B), B~Beta(1/2,(d-1)/2): psi((d-1)/2)-psi(d/2)
for d in [10, 100, 1000, 10**4]:
    print(f"d={d}: d*E ln(1-B) = {d*(special.digamma((d-1)/2)-special.digamma(d/2)):.6f}, d*ln(1-1/d)={d*np.log1p(-1/d):.6f}")
# Kaczmarz simulation: angular error eps, real registers vs b-bit quantized; T_needed/(d ln(1/eps^2))
def kacz(d, eps, qbits, reps=60, Tmax_factor=1.6):
    T0 = int(Tmax_factor * d * np.log(1 / eps**2)) + 5 * d
    hits = []
    for r in range(reps):
        s = rng.standard_normal(d); s /= np.linalg.norm(s)
        z = np.zeros(d); tt = None
        for t in range(1, T0 + 1):
            x = rng.standard_normal(d); y = x @ s
            z = z + (y - x @ z) / (x @ x) * x
            if qbits is not None:
                z = np.round(z * 2.0**qbits) / 2.0**qbits     # fixed point, q fractional bits per coord
            nz = np.linalg.norm(z)
            if nz > 0 and np.arccos(np.clip(z @ s / nz, -1, 1)) <= eps:
                tt = t; break
        hits.append(tt if tt is not None else np.inf)
    hits = np.array(hits)
    return np.quantile(hits, 2/3) / (d * np.log(1 / eps**2)), np.median(hits) / (d * np.log(1 / eps**2))
for (d, eps) in [(200, 0.1), (200, 1e-3), (400, 1e-4)]:
    lam = np.log2(1/eps)
    for q in [None, int(np.ceil(lam + np.log2(d) + 4)), int(np.ceil(lam - 2))]:
        r = kacz(d, eps, q, reps=30)
        print(f"Kaczmarz d={d} eps={eps} qbits={q}: first-hit T(2/3-quantile)/(d ln eps^-2)={r[0]:.4f}, median ratio={r[1]:.4f}")
