#!/bin/sh
# Reproduce all computations of paper/main.tex.
# Requirements: python3 with mpmath, numpy, scipy (tested with Python 3.10, mpmath 1.3.0,
# numpy 2.2.6, scipy 1.15.3).  Single-threaded unless stated; times are wall-clock on one core.
# Usage: ./run_all.sh          (everything, about 20 minutes)
#        ./run_all.sh quick    (skips the three slowest steps, about 4 minutes)
# Reference outputs are in expected_output/.  Monte Carlo steps use fixed seeds; their last
# digits may differ across numpy versions.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
mode="${1:-full}"
step() { echo; echo "== $1"; }

step "certify.py: interval certificate 1.38470 (Lemma 11), ~1 s, mpmath";            python3 certify.py
step "ledger.py: loss ledger (Table 1), ~1 s, stdlib";                                 python3 ledger.py
step "rates.py: rate functions, framework limit 0.333571, ~20 s, numpy+scipy";         python3 rates.py
step "checks.py: beta laws, moving-split domination, cap, Kaczmarz, ~15 s, numpy+scipy"; python3 checks.py
step "finite_d.py 1e6 1e-3 2e-2 1e-4: auxiliary finite-d evaluation, ~1 s, stdlib";    python3 finite_d.py 1e6 1e-3 2e-2 1e-4
step "finite_d_fullscan.py: full (b,j) scan for Table 2, ~25 s, numpy+scipy";          python3 finite_d_fullscan.py
step "independent/indep_mixture.py: re-implementation, mixture lemma, ~25 s, numpy+scipy+mpmath"; python3 independent/indep_mixture.py
step "independent/indep_constant.py: re-implementation, constant and finite-d grid, ~25 s, numpy+mpmath"; python3 independent/indep_constant.py
step "memratio.py: Theta(d^2) remark, ~1 min, numpy+scipy";                            python3 memratio.py
step "independent/indep_cap_kacz.py: arcsin cap D=2..399 and Kaczmarz, ~75 s, numpy+scipy+mpmath"; python3 independent/indep_cap_kacz.py
if [ "$mode" = quick ]; then
  echo; echo "quick mode: skipped inverse_moment_check.py (~1.5 min), finite_d_fine.py (~3.5 min), kacz_fixedpoint.py (~10-17 min)"
  exit 0
fi
step "inverse_moment_check.py: inverse-moment lemma (Lemma 9), ~1.5 min, mpmath";      python3 inverse_moment_check.py
step "finite_d_fine.py: finite-d table (Table 2), ~3.5 min, numpy+mpmath";              python3 finite_d_fine.py
step "kacz_fixedpoint.py: fixed-point Kaczmarz unit test (Table 3), ~10-17 min, numpy (4 BLAS threads)"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 kacz_fixedpoint.py
