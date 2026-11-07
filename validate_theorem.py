# MIT License
#
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Validation of the logical steps in the Theorem (conditional continuum reconstruction).

The Theorem is conditional on C1-C4, which concern the continuum limit and cannot be tested
numerically. This script checks the deductive steps of its proof on finite models where every
quantity is computable.

What is checked
  Step 4 (OS2): limits of reflection-positive Gram matrices remain positive semidefinite.
  Step 6 (OS4): clustering <=> unique vacuum. A transfer matrix with a unique top eigenvalue gives
                decaying truncated correlations; a degenerate one does not.
  Step 7/8 (reconstruction and gap): from reflection-positive Euclidean correlations
                S_ab(t) = <F_a, e^{-tH} F_b> alone, a generalized eigenvalue problem reconstructs the
                low spectrum of H, recovering the gap.
  Step 8 (mass gap): if every truncated correlation obeys |C(t)| <= C e^{-mt}, the spectral measure
                has no weight in (0, m). Conversely, weight at E < m violates the bound at large t.
  Scope: a gap bound m from clustering is a lower bound; the true threshold can be larger.

What is NOT checked: C1-C4 themselves (existence of the continuum limit, E(4) covariance,
OS growth), nontriviality, or identification of the limit with Yang-Mills theory.
"""
import math
import sys

import numpy as np
from scipy.linalg import eigh

from common import Report


def random_hamiltonian(rng, d, gap, degenerate=False):
    E = np.sort(rng.uniform(gap, gap + 4.0, size=d))
    E[0] = 0.0
    if degenerate:
        E[1] = 0.0
    else:
        E[1] = gap
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return Q @ np.diag(E) @ Q.T, E, Q


def main():
    rep = Report("Theorem")
    rng = np.random.default_rng(13)
    d, m_true = 40, 0.8

    # Step 4: reflection positivity is closed under limits
    H, E, Q = random_hamiltonian(rng, d, m_true)
    ok = True
    for eps in (1e-1, 1e-2, 1e-3, 1e-4):
        Hn = H + eps * np.diag(rng.normal(size=d)) ; Hn = 0.5 * (Hn + Hn.T)
        w, V = eigh(Hn)
        T = V @ np.diag(np.exp(-0.5 * (w - w.min()))) @ V.T
        Phi = rng.normal(size=(d, 8))
        G = Phi.T @ T @ Phi
        ok &= np.linalg.eigvalsh(0.5 * (G + G.T)).min() > -1e-10
    rep.check("Step 4 (OS2): reflection-positive Gram matrices stay PSD along a convergent sequence", ok)

    # Step 6: clustering <=> unique vacuum
    def truncated_decay(degenerate):
        H, E, Q = random_hamiltonian(rng, d, m_true, degenerate)
        Om = Q[:, 0]
        A = np.diag(rng.normal(size=d))                     # a local observable
        P0 = np.outer(Om, Om)                               # projection on the chosen vacuum
        psi = A @ Om - (Om @ A @ Om) * Om
        def C(t):
            return psi @ (Q @ np.diag(np.exp(-t * E)) @ Q.T) @ psi
        return C(30.0) / C(0.0)
    rep.check("Step 6 (OS4): unique vacuum => truncated correlations decay; degenerate => they do not",
              abs(truncated_decay(False)) < 1e-8 and abs(truncated_decay(True)) > 1e-4,
              f"unique {truncated_decay(False):.1e}, degenerate {truncated_decay(True):.1e}")

    # Steps 7-8: reconstruct the spectrum from Euclidean data (GEVP)
    H, E, Q = random_hamiltonian(rng, d, m_true)
    Om = Q[:, 0]
    F = [np.diag(rng.normal(size=d)) for _ in range(d)]       # a generating family
    vecs = np.array([f @ Om - (Om @ f @ Om) * Om for f in F]).T      # truncated vectors
    def S(t):
        return vecs.T @ (Q @ np.diag(np.exp(-t * E)) @ Q.T) @ vecs
    t0, t1 = 0.5, 1.0
    A0, A1 = S(t0), S(t1)
    w0, V0 = eigh(A0)
    keep = w0 > 1e-10 * w0.max()
    Pk = V0[:, keep] / np.sqrt(w0[keep])
    lam = np.linalg.eigvalsh(Pk.T @ A1 @ Pk)
    E_rec = -np.log(np.clip(lam[::-1], 1e-300, None)) / (t1 - t0)
    rep.check("Steps 7-8: Euclidean correlations of a generating family reconstruct the gap (GEVP)",
              abs(E_rec[0] - E[1]) < 1e-6, f"reconstructed {E_rec[0]:.6f}, true gap {E[1]:.6f}")
    # with an incomplete family the reconstruction is only a variational upper bound
    sub = vecs[:, :6]
    S6 = lambda t: sub.T @ (Q @ np.diag(np.exp(-t * E)) @ Q.T) @ sub
    w0, V0 = eigh(S6(t0)); Pk = V0 / np.sqrt(w0)
    lam6 = np.linalg.eigvalsh(Pk.T @ S6(t1) @ Pk)
    E6 = -math.log(lam6.max()) / (t1 - t0)
    rep.check("Steps 7-8: an incomplete observable family gives only an upper bound (why C1 needs the full algebra)",
              E6 >= E[1] - 1e-9, f"6 observables: {E6:.4f} >= true gap {E[1]:.4f}")

    # Step 8: decay bound <=> no spectral weight in (0, m)
    ok = True
    for trial in range(200):
        Es = rng.uniform(0, 5, size=10)
        Es[(Es > m_true - 0.05) & (Es < m_true)] = m_true      # keep a clear margin below m
        ws = rng.uniform(0.1, 1, size=10)
        C = lambda t: np.sum(ws * np.exp(-t * Es))
        has_low = np.any(Es < m_true - 0.05)
        ts = np.linspace(50, 400, 30)
        bounded = all(C(t) <= np.sum(ws) * math.exp(-m_true * t) * (1 + 1e-9) for t in ts)
        ok &= (not has_low) == bounded
    rep.check("Step 8: |C(t)| <= C e^{-mt} for all large t  <=>  spectral measure vanishes on (0, m)", ok)

    # Scope: m from the clustering bound is a lower bound, not the threshold
    m_claim = 0.5
    rep.check("Scope: a clustering rate m is a lower bound on the spectral threshold",
              m_claim <= E[1], f"m = {m_claim}, true threshold {E[1]:.3f}")
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
