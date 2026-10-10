# The leading constant of the memory–sample tradeoff for noiseless Gaussian regression

A streaming learner reads exact Gaussian linear equations `Y_t = ⟨X_t, s⟩`, `X_t ~ N(0, I_d)`.
It keeps at most `M` bits between samples and must output a unit vector within angle `ε` of `s`.
OpenAI's preprint *Subsphere methods for memory–sample lower bounds in noiseless Gaussian
regression* (September 27, 2026) shows that `M = o(d²)` forces `T ≥ 2^{-16} d log₂(1/ε)`
samples. This repository contains a note and the computations behind it. They show that
**the same conclusion holds with `2^{-16}` replaced by any constant `c < 2 ln 2 ≈ 1.386`**,
uniformly in `0 < ε ≤ 1/10`. They also show that **randomized Kaczmarz with fixed-point
registers matches this constant**. So in the memory window `d(log₂(1/ε) + 2 log₂ d) ≤ M = o(d²)`
the optimal sample count is `(2 ln 2 + o(1)) d log₂(1/ε)`. The improvement over the upstream
constant is a factor of about `9.07 × 10⁴`. It comes from three changed estimates, and the
first of them is needed: with the upstream fixed split at `1/2`, no choice of parameters gives
more than `0.5176`.

The results were found and checked by **Claude (Anthropic) agents** working under the
direction of the repository owner, [AUTHOR NAME]. No human peer review has taken place, and
nothing here is formalized in Lean. The results depend on the upstream preprint being
correct. See [Status and caveats](#status-and-caveats).

**Note:** [*The leading constant of the memory–sample tradeoff for noiseless Gaussian
regression*](paper/main.pdf) ([LaTeX source](paper/main.tex), [build
instructions](paper/README.md)), 12 pages.

## What is proved

Numbering follows `paper/main.pdf`. `λ = log₂(1/ε)` and `λ' = log₂(1/sin ε)`.

| Statement | Result | Kind of proof |
| --- | --- | --- |
| Upstream, Theorem 1.2 | `M = o(d²)` ⇒ `T ≥ 2^{-16} d λ` for large `d` | upstream |
| **Theorem 2** (lower bound) | `M = o(d²)`, any `c < 2 ln 2`: `T ≥ c d λ' ≥ c d λ` for `d ≥ d₀(c, M)`, uniformly in `0 < ε ≤ 1/10`, uniform prior, success `≥ 2/3` | note, by hand; numerical checks |
| **Lemma 3** (finite-bit Kaczmarz) | for `d ≥ 16/η²`: `M = d(q+3) ≤ d(λ + (3/2)log₂ d + log₂(1/η) + 7.3)` bits and `T = ⌈(1+η) d ln(1/sin²ε)⌉` samples succeed for **every** signal | note, by hand; unit test (Table 3) |
| **Corollary 4** (sharpness window) | if `d(λ + 2 log₂ d) ≤ M = o(d²)`, the optimal horizon `T*` satisfies `T* / (d ln(1/sin²ε)) → 1`, i.e. `T* = (2 ln 2 + o(1)) d λ` when `ε → 0` | Theorem 2 + Lemma 3 |
| **Lemma 11** | `sup Φ = 2 ln 2`; `Φ(10⁻⁷, 10⁻⁴, 10⁻⁸, 10⁻³) ≥ 1.38470` | by hand; 200-bit interval arithmetic |
| **§5.1** (limit of the upstream method) | with the split fixed at `1/2`: at most `0.51759 = 2 ln 2 / u'`; with the upstream ratio estimates as well: at most `0.33357 = ln 2 / u` (`u, u'` Lambert-W values) | closed form; numerical optimization agrees |
| **Table 2** (finite `d`) | effective constant `0.341, 0.779, 1.071, 1.231` at `d = 10⁴, 10⁶, 10⁸, 10¹⁰` (`λ = 10`) | numerical **estimates**, not certificates |

The three changed estimates are a moving split point `q₀ = 1 − δ` in the beta-mixture
domination (Lemma 7), an exact beta-ratio bound (Lemma 8) and an `arcsin` form of the terminal
cap bound (Lemma 6). Appendix A lists the line-level changes to the upstream proof.

### Where the constant was lost

Paper parameters `k = m = ⌊d/16⌋`, `ℓ = ⌊d/32⌋`, `d = 4·10⁴`, `λ → ∞` (Table 1,
`computations/ledger.py`):

| Step | Change | Constant `c` | Factor |
| --- | --- | --- | --- |
| S0 | upstream as printed | `2^{-16} = 1.53·10⁻⁵` | – |
| S1 | same displayed bounds without arithmetic slack | `2^{-10}` | ×64 |
| S2 | exact values of the stated lemma bounds | `3.88·10⁻³` | ×4.0 |
| S3 | the constant the upstream proof actually yields (`2^{(b−1)/2}`, not `2^{D'}`) | `0.1048` | ×27 |
| S3b | split at `1/2`, exact beta ratios | `0.1195` | ×1.14 |
| S4 | moving split `q₀ = 1 − δ` (Lemma 7) | `0.8704` | ×7.3 |
| S5 | terminal cap `8 → 1` (Lemma 6); matters for bounded `λ` (`0.0655 → 0.8515` at `λ = log₂ 10`) | `0.8704` | – |
| S6 | parameters `μ ≪ κ ≪ π → 0` (Lemma 11) | `→ 2 ln 2 = 1.3863` | ×1.59 |
| | certified uniform constant | `1.38470` | total ×9.07·10⁴ |

The largest losses are the arithmetic slack (S1), the gap between the statement and the proof
of the upstream mixture lemma (S3), and the fixed split (S4). Only S4 needs a new estimate. The
fixed split costs `(κ/2) ln 2` nats per unit dimension per block whatever the parameters are,
and no choice of parameters removes it.

## Repository layout

- `paper/`: LaTeX source and PDF of the note. `paper/build.sh` builds it (pdflatex only).
- `computations/`: Python scripts for every number in the note. `computations/run_all.sh`
  runs all of them (about 20 minutes on one core; `./run_all.sh quick` takes about 4 minutes
  and skips the three slowest steps). It needs `python3`, `mpmath`, `numpy` and `scipy`.
  - `certify.py`: interval certificate of `1.38470` (Lemma 11).
  - `ledger.py`, `costs.py`: the loss ledger (Table 1) and exact per-block cost functions.
  - `rates.py`: `d → ∞` rate functions, and the upstream-framework limit `0.333571` (§5.1).
  - `checks.py`: Monte Carlo of the beta laws, pointwise moving-split domination, terminal cap, Kaczmarz.
  - `inverse_moment_check.py`: Lemma 9, including the counterexample to the tempting bound.
  - `finite_d_fine.py`, `finite_d_fullscan.py`, `finite_d.py`: finite-`d` evaluation (Table 2).
    The full scan over all batch lengths `b` and directions `j` at the reported optimizers
    gives the same block maximum as the sampled points.
  - `memratio.py`: the `Θ(d²)`-memory remark.
  - `kacz_fixedpoint.py`: unit test of Lemma 3 (Table 3).
  - `independent/`: separate re-implementations of the mixture lemma, the cap lemma, the
    constant and the finite-`d` grid. They do not import the scripts above.
  - `expected_output/`: reference output of every step.

## Status and caveats

- **Asymptotic.** Theorem 2 and Corollary 4 are statements about `d → ∞`. The convergence is
  slow. With Kaczmarz memory `M = d(λ + 2 log₂ d)` and `λ = 10`, the effective lower-bound
  constant is about:

  | `d` | `10⁴` | `10⁶` | `10⁸` | `10¹⁰` |
  | --- | --- | --- | --- | --- |
  | effective constant (estimate) | 0.341 | 0.779 | 1.071 | 1.231 |

  Compare `2 ln 2 = 1.386` and the upstream `2^{-16}`. These values are estimates of what the
  proof gives and not proved bounds: the block maximum is evaluated on sampled `(b, j)`. A full
  scan at `d = 10⁴, 10⁶` agrees with the sampled points. At `d = 10⁴` the gap to Kaczmarz is
  still about a factor of 4, and at `d = 10¹⁰` it is 11%. "Kaczmarz is first-order optimal" is
  an asymptotic statement.
- **`Θ(d²)` memory is open.** For `M = A d²`, Theorem 2 is not available. A heuristic
  computation gives only `c(A) ≈ 0.47, 0.72, 0.93, 1.19, 1.31` for
  `A = 10⁻², 10⁻³, 10⁻⁴, 10⁻⁶, 10⁻⁸`. The sharp constant for `Θ(d²)` memory is unknown.
- **The window matters.** The upper bound needs `M ≥ d(λ + 2 log₂ d)`. Below `d λ` bits it does
  not apply.
- **Conditional on upstream.** The architecture of the lower bound and every unchanged lemma
  come from the upstream preprint. We checked the steps we changed, and found no step of the
  upstream proof to be wrong. We have not independently re-verified the rest. If the upstream
  proof has a gap, so do Theorem 2 and Corollary 4.
- **Does not answer Sharan–Sidford–Valiant (STOC 2019).** Their question concerns the order of
  the sample complexity with noisy labels and `d²/4` bits. The companion OpenAI preprint
  *Memory and precision in noiseless Gaussian regression* addresses the order. This note only
  sharpens `Θ(d log(1/ε))` to the leading constant `2 ln 2` for `o(d²)` memory.
- **Lemma 3 is not new except for its constants.** Sharan, Sidford and Valiant remark without
  proof that randomized Kaczmarz achieves `O(d log(1/ε))` samples and memory. Lemma 3 writes
  this out with explicit constants. The `3/2 log₂ d` term was not optimized.
- **Numerical evidence.** Interval arithmetic certifies the constant in Lemma 11. The other
  scripts are checks (Monte Carlo, grids, unit tests), not proofs.
- **Citations.** The bibliographic data for Sharan–Sidford–Valiant (STOC 2019, pp. 890–901, DOI
  and arXiv number), Marsden–Sharan–Sidford–Valiant (PMLR 178:2390–2430), Blanchard–Zhang–Jaillet
  (PMLR 195:4696–4736) and Strohmer–Vershynin (J. Fourier Anal. Appl. 15, 262–278) was checked
  against the publishers' pages. The Kaczmarz (1937) entry is the standard citation and was not
  checked against the original. Quotations from the OpenAI preprints refer to their sources at
  the commit in [`UPSTREAM.md`](UPSTREAM.md).
- **No human peer review, not formalized.** No historical-priority claim is made beyond "not
  found in the literature we were able to read".

## AI attribution

The results in this repository were found, verified and written up by Claude (Anthropic)
agents working under the direction of the repository owner, [AUTHOR NAME]. This covers the
analysis, the note and the computations, including the independent re-implementations in
`computations/independent/`. The human owner directed the work and is responsible for its
release.

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite the OpenAI preprints listed in
[`UPSTREAM.md`](UPSTREAM.md).
