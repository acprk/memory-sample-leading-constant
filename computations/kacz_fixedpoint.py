"""Unit test for Lemma "Finite-bit randomized Kaczmarz" in paper/main.tex.

Algorithm (exactly the one in the lemma):
  state z in G_q^d, G_q = 2^{-q} Z cap [-3,3];  z_0 = 0
  read the real pair (x_t, y_t), form  zt = z + (y_t - <x_t,z>)/|x_t|^2 x_t  from exact (float64) data,
  store z <- round_q(zt)  (nearest point of 2^{-q}Z; the lemma proves |zt|_inf <= 3 so no clipping occurs)
  after T samples output z/|z|.
Lemma parameters:  T = ceil((1+eta) d ln(1/sin^2 eps)),
                   q = ceil(log2(1/sin eps) + 1.5 log2 d + log2(4 sqrt6 / eta)),  M = d (q+3) bits.
Checked per configuration (R independent signals, all trials vectorised):
  succ_fix   : fraction with arccos<z/|z|, s> <= eps at time T   (lemma: >= 2/3; Chebyshev/Markov proof is loose)
  succ_real  : same for real (float64) registers (no rounding)
  rms|a_T|   : rounding accumulation a_T = e_T - ehat_T  vs proven L2 bound 2 d^{3/2} 2^{-q-1}
  P(sum B<L) : empirical lower-tail probability of the B-sum (proof: <= 11.52/(eta^2 T))
  max|z|_inf : must be <= 3 (deterministic range claim)
  q_low      : the same run with q = ceil(log2(1/sin eps)) - 3 bits per coordinate (expected to fail; shows
               the lambda term is necessary) ;  first-hit ratio: median over trials of first t with
               angular error <= eps (fixed point, lemma q), divided by d ln(1/sin^2 eps).
usage: python3 kacz_fixedpoint.py [R]      (single process, BLAS threads <= 4)
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np
from math import log, log2, ceil, sin, sqrt

R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(20261009)


def run(d, eps, eta, q, R, track=True):
    se = sin(eps)
    ell = log(1 / se**2)
    T = ceil((1 + eta) * d * ell)
    S = rng.standard_normal((R, d)); S /= np.linalg.norm(S, axis=1, keepdims=True)
    z = np.zeros((R, d))          # fixed-point state
    zr = np.zeros((R, d))         # exact Kaczmarz (real registers), same rows
    scale = 2.0 ** q if q is not None else None
    sumB = np.zeros(R); zmax = 0.0
    first = np.full(R, np.inf)
    cos_eps = np.cos(eps)
    for t in range(1, T + 1):
        X = rng.standard_normal((R, d))
        Y = np.einsum("ij,ij->i", X, S)
        nx2 = np.einsum("ij,ij->i", X, X)
        if track:
            er = zr - S
            ner2 = np.einsum("ij,ij->i", er, er)
            sumB += np.einsum("ij,ij->i", X, er) ** 2 / (nx2 * np.maximum(ner2, 1e-300))
        zr += ((Y - np.einsum("ij,ij->i", X, zr)) / nx2)[:, None] * X
        zt = z + ((Y - np.einsum("ij,ij->i", X, z)) / nx2)[:, None] * X
        if scale is not None:
            zmax = max(zmax, float(np.abs(zt).max()))
            z = np.round(zt * scale) / scale
        else:
            z = zt
        if track and (t % 50 == 0 or t == T):
            nz = np.linalg.norm(z, axis=1)
            ok = np.einsum("ij,ij->i", z, S) >= cos_eps * nz
            newly = ok & ~np.isfinite(first)
            first[newly] = t
    def succ(Z):
        nz = np.linalg.norm(Z, axis=1)
        return float(np.mean((nz > 0) & (np.einsum("ij,ij->i", Z, S) >= cos_eps * nz)))
    out = dict(T=T, succ_fix=succ(z), succ_real=succ(zr), zmax=zmax)
    if track:
        a = z - zr   # = e_T - ehat_T
        out["rms_a"] = float(np.sqrt(np.mean(np.sum(a * a, axis=1))))
        L = ell + 2 * log(1 / (1 - eta / 4))
        out["P_sumB_lt_L"] = float(np.mean(sumB < L))
        out["first_ratio_med"] = float(np.median(first) / (d * ell))
        out["first_ratio_q23"] = float(np.quantile(first, 2 / 3) / (d * ell))
    return out


if __name__ == "__main__":
    eta = 0.25
    print(f"R={R} trials per row, eta={eta}; Chebyshev threshold of the lemma d >= 16/eta^2 = {16/eta**2:.0f}")
    hdr = ("d", "eps", "lam", "q", "M/d", "T", "succ_fix", "succ_real", "rms|a_T|", "L2 bound", "P(sumB<L)",
           "max|zt|", "1st-hit med", "1st-hit q2/3", "q_low", "succ(q_low)")
    print(" | ".join(hdr))
    for d, eps in [(200, 0.1), (200, 1e-3), (200, 1e-5), (400, 0.1), (400, 1e-3), (400, 1e-4),
                   (1000, 0.1), (1000, 1e-3)]:
        t0 = time.time()
        se = sin(eps)
        q = ceil(log2(1 / se) + 1.5 * log2(d) + log2(4 * sqrt(6) / eta))
        o = run(d, eps, eta, q, R)
        qlow = max(1, ceil(log2(1 / se)) - 3)
        o2 = run(d, eps, eta, qlow, R, track=False)
        bound = 2 * d * sqrt(d) * 2.0 ** (-q - 1)
        print(f"{d} | {eps:g} | {log2(1/eps):.2f} | {q} | {q+3} | {o['T']} | {o['succ_fix']:.3f} | {o['succ_real']:.3f} | "
              f"{o['rms_a']:.3e} | {bound:.3e} | {o['P_sumB_lt_L']:.3f} | {o['zmax']:.3f} | {o['first_ratio_med']:.4f} | "
              f"{o['first_ratio_q23']:.4f} | {qlow} | {o2['succ_fix']:.3f}    [{time.time()-t0:.0f}s]", flush=True)
