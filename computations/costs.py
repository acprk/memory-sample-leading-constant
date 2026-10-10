"""Exact per-block cost functions for the beta-mixture proof of
the OpenAI preprint "Subsphere methods for memory-sample lower bounds in noiseless
Gaussian regression" (build/sections/07-beta-mixture.tex).

All quantities are natural logs.  Notation follows the paper:
  D   = current test dimension, b = batch length, n = D-b, D' = D-1,
  j   = number of earlier replica directions (0..m-1),
  l   = integer inverse-radius exponent, p = 2l.
Mixture lemma (Lemma s:mixture):
  alpha=(n-j)/2, alpha*=(D'-j)/2, beta0=j/2, beta1=(D'-n+j-1)/2=(b+j-2)/2,
  Law(v) <= A(q0) nu0 + B(q0) nu1  with  (split at q0; the paper uses q0=1/2)
  A(q0) = B(a*,b0)/B(a,b0) * q0^{-(a*-a)},  B(q0) = B(a,b1)/B(a,b0) * (1-q0)^{-(b1-b0)}
  factor_j = A + B * E Z^{-l},  Z ~ Beta(alpha, alpha*-alpha)
Residual: E rho^{-p}, rho^2 ~ Beta(n/2, b/2).
"""
from math import lgamma, log, exp, log1p, inf

def lB(a, b):
    return lgamma(a) + lgamma(b) - lgamma(a + b)

def ln_inv_moment(A, Atot, l):
    """ln E Z^{-l} for Z~Beta(A, Atot-A), integer l < A:  prod_{t=1}^l (Atot-t)/(A-t)."""
    if l >= A:
        return inf
    return lgamma(Atot) + lgamma(A - l) - lgamma(A) - lgamma(Atot - l)

def ln_residual(D, b, l):
    return ln_inv_moment((D - b) / 2, D / 2, l)

def logaddexp(x, y):
    if x < y:
        x, y = y, x
    if y == -inf:
        return x
    return x + log1p(exp(y - x))

def mix_terms(D, b, j, l):
    Dp = D - 1; n = D - b
    a = (n - j) / 2; a_s = (Dp - j) / 2; b0 = j / 2; b1 = (b + j - 2) / 2
    lA0 = lB(a_s, b0) - lB(a, b0)          # <= 0
    lB0 = lB(a, b1) - lB(a, b0) if b1 > b0 else 0.0
    lEZ = ln_inv_moment(a, a_s, l)
    return a_s - a, b1 - b0, lA0, lB0, lEZ

def ln_factor(D, b, j, l, mode="opt"):
    """ln of the mixture factor for direction j+1.
    mode: 'paper'  : 2^{D'} (1 + E Z^{-l})        (statement of Lemma s:mixture)
          'half'   : 2^{(b-1)/2}(1+E Z^{-l})      (what the proof at q0=1/2 gives)
          'half_ex': exact A(1/2)+B(1/2)E Z^{-l}  (split at 1/2, keep Beta ratios)
          'opt'    : min_{q0} A(q0)+B(q0) E Z^{-l} (new: moving split)"""
    if j == 0 or b == 1:
        return 0.0
    eA, eB, lA0, lB0, lEZ = mix_terms(D, b, j, l)
    if mode == "paper":
        return (D - 1) * log(2) + logaddexp(0.0, lEZ)
    if mode == "half":
        return eA * log(2) + logaddexp(0.0, lEZ)
    def val(u):  # u = -ln(1-q0) > 0 ; q0 = 1-e^{-u}
        lq0 = log1p(-exp(-u)) if u < 30 else -exp(-u)
        return logaddexp(lA0 - eA * lq0, lB0 + eB * u + lEZ)
    if mode == "half_ex":
        return val(log(2))
    # golden-section on u in log scale; function is unimodal (A decreasing, B increasing in u)
    lo, hi = -25.0, 6.0   # log u range
    g = (5 ** 0.5 - 1) / 2
    x1 = hi - g * (hi - lo); x2 = lo + g * (hi - lo)
    f1 = val(exp(x1)); f2 = val(exp(x2))
    for _ in range(120):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo); f1 = val(exp(x1))
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo); f2 = val(exp(x2))
    return min(f1, f2, val(log(2)))

def ln_block(D, b, m, l, mode="opt", jstep=1):
    """ln of per-block factor without the w^{1/m} term:
       (1/m) sum_{j<m} ln factor_j + ln E rho^{-p}.  (paper: worst j for all m)"""
    if mode == "paper_worst":
        lf = max(ln_factor(D, b, j, l, "paper") for j in (1, m - 1))
        return lf + ln_residual(D, b, l)
    tot = 0.0; cnt = 0
    js = list(range(0, m, jstep))
    for j in js:
        tot += ln_factor(D, b, j, l, mode); cnt += 1
    return tot / cnt + ln_residual(D, b, l)

def T_lower(d, k, l, lnG, lam, lnCterm):
    """Contradiction condition: success <= G^J (C eps)^p, J <= T/k + 1, success >= 2/3.
    Returns smallest T not excluded:  T >= k[(p(lam ln2 - lnC) - ln1.5)/lnG - 1]."""
    p = 2 * l
    return k * ((p * (lam * log(2) - lnCterm) - log(1.5)) / lnG - 1)
