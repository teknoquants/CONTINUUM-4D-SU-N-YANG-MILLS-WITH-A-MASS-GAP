# Validation scripts: Continuum 4D SU(N) Yang–Mills with a Mass Gap

Companion code to the white paper by R. Cooperman and A.E. Petrunina V.V. (XISOSIS, Vienna, 7 November 2026).
Released under the MIT License (see `LICENSE`; the licence is also reproduced at the top of every script).

GitHub: [GITHUB REPOSITORY URL — placeholder] · DOI: [DOI — placeholder]

## Run

```
pip install numpy scipy
python run_all.py
```

Requires Python 3.10+, NumPy ≥ 2.0 and SciPy ≥ 1.9 (for `scipy.optimize.milp`). Runtime is about 10 seconds.

## What each script validates

| Script | Paper item | Checks |
|---|---|---|
| `validate_lemma1.py` | Lemma 1 | Threshold algebra for g₀(N); plaquette degree 12; lattice-animal bound (12e)ⁿ; at most 8(r+s) plaquettes touch C; **area count by exact integer programming** (ℤ₂ and ℤ₃); summation bound; **center selection rule** (SU(2) Monte Carlo); illustration of the Yarotsky gap mechanism |
| `validate_lemma2.py` | Lemma 2 | Positivity and reflection positivity of the transfer matrix; second-order norm convergence of the product formula; ground-state convergence; correlation limit; decay rates ≥ δ with infimum = δ (two coupled rotors) |
| `validate_lemma3.py` | Lemma 3 | SU(2) heat kernel normalization and character eigenvalues; gluing K_s∗K_t = K_{s+t}; blocking to K_{b²t}; area law exp(−t c_N\|A\|/2); step count k |
| `validate_lemma4_5.py` | Lemmas 4, 5 | b₁ value; unbounded one-loop/two-loop discrepancy (why a(g) is two-loop); conclusions of both lemmas on 4000 synthetic RG data sets satisfying all hypotheses; failure without 𝔊₄ |
| `validate_theorem.py` | Theorem | Reflection positivity under limits; clustering ⇔ unique vacuum; gap reconstruction from Euclidean data (GEVP); incomplete observable families give only upper bounds; decay bound ⇔ no spectral weight in (0, m) |

## What is not validated

The open hypotheses of the paper cannot be checked numerically: the certified Yarotsky constants and link-embedding estimate (𝔊₁), the expansion 𝔈(g, K), the renormalization-group hypotheses 𝖱, 𝖱′, 𝖸, 𝖸′, 𝖠𝖥, 𝖠𝖥±, 𝔊₄, 𝔊₅, and the continuum hypotheses 𝖢₁–𝖢₄. Lemmas 2 and the Theorem are checked on finite-dimensional models with the same operator structure, not on four-dimensional Yang–Mills theory. Passing these checks therefore confirms the proved statements and the logic of each implication, not the Millennium Problem.
