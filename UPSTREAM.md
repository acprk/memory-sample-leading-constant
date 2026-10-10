# Upstream provenance

This repository contains no code or text copied from other repositories. It analyses two
preprints in [OpenAI's mathematics repository](https://github.com/openai/math), at commit
[`fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb`](https://github.com/openai/math/tree/fd4aeeb2ee4fc729c18d98444fed42fd0529eeeb/preprints):

- *Subsphere methods for memory–sample lower bounds in noiseless Gaussian regression* (OpenAI,
  September 27, 2026), directory
  `preprints/Subsphere-methods-for-memory-sample-lower-bounds-in-noiseless-Gaussian-regression-September-27-2026`.
  It proves the lower bound `T ≥ 2^{-16} d log₂(1/ε)` for `o(d²)` memory (Theorem 1.2).
- *Memory and precision in noiseless Gaussian regression* (OpenAI, September 27, 2026),
  directory `preprints/Memory-and-precision-in-noiseless-Gaussian-regression-September-27-2026`.
  The note uses it only for the model, for the real-register Kaczmarz identity
  `E‖z_t − s‖² = (1 − 1/d)^t` and for the context of the order question.

The architecture of the lower bound and every unchanged lemma are from the first preprint. This
repository changes three estimates in it and re-optimizes the parameters. File and line
references in the note (`02-subspace-moments.tex`, `07-beta-mixture.tex`, ...) refer to
`build/sections/` of the first preprint, and `introduction.tex` to `build/sections/` of the
second, at the commit above.
