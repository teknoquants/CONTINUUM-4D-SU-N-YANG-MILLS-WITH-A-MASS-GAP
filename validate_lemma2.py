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
"""Validation of Lemma 2 (Hamiltonian-Euclidean bridge) on a finite-volume model.

Model: two coupled quantum rotors (the abelian analogue of the link variables) in a truncated
Fourier basis, H = (g^2/2) K + V with K = n1^2 + n2^2 (Laplacian, elliptic) and the bounded
potential V = g^-2 (1 - cos(theta1 - theta2)). This has the structure used in the proof:
nonnegative self-adjoint K, smooth bounded V, a unique positive ground state.

What is checked
  Gate:  T_tau = e^{-tau V/2} e^{-tau (g^2/2) K} e^{-tau V/2} is positive semidefinite, and the
         reflection Gram matrices <phi_a, T^j phi_b> are positive semidefinite.
  (i)    Norm convergence of the symmetric product formula T_tau^{t/tau} -> e^{-tH}, with the
         second-order rate, convergence of the top eigenvector of T_tau to the ground state, and
         convergence of time-separated correlations to <Omega, F e^{-t(H-E0)} G Omega>.
  (ii)   Every truncated correlation of a multiplication operator decays at rate >= delta, and the
         infimum over observables equals the gap delta.

What is NOT checked: uniformity in volume L or coupling g (the paper asserts none), and transfer of
Balaban or Shen-Zhu-Zhu estimates to the Hamiltonian.
"""
import math
import sys

import numpy as np
from scipy.linalg import expm, eigh

from common import Report

M = 8          # Fourier cut-off |n_i| <= M
g = 1.1
t_ref = 1.0


def build():
    ns = np.arange(-M, M + 1)
    idx = {(a, b): k for k, (a, b) in enumerate((a, b) for a in ns for b in ns)}
    d = len(idx)
    K = np.zeros((d, d))
    C = np.zeros((d, d))      # cos(theta1 - theta2): (n1, n2) -> (n1 +/- 1, n2 -/+ 1)
    def mult(shifts):
        X = np.zeros((d, d), dtype=complex)
        for (a, b), k in idx.items():
            for (da, db), c in shifts:
                key = (a + da, b + db)
                if key in idx:
                    X[idx[key], k] += c
        return X
    for (a, b), k in idx.items():
        K[k, k] = a * a + b * b
    C = np.real(mult([((1, -1), 0.5), ((-1, 1), 0.5)]))
    V = (np.eye(d) - C) / g ** 2
    H = 0.5 * g ** 2 * K + V
    obs = {
        "cos th1": mult([((1, 0), 0.5), ((-1, 0), 0.5)]),
        "sin th1": mult([((1, 0), -0.5j), ((-1, 0), 0.5j)]),
        "cos(th1-th2)": mult([((1, -1), 0.5), ((-1, 1), 0.5)]),
        "cos 2th1": mult([((2, 0), 0.5), ((-2, 0), 0.5)]),
        "sin(th1+th2)": mult([((1, 1), -0.5j), ((-1, -1), 0.5j)]),
    }
    return K, V, H, obs


def main():
    rep = Report("Lemma 2")
    rng = np.random.default_rng(7)
    K, V, H, obs = build()
    E, Uv = eigh(H)
    Om = Uv[:, 0]
    delta = E[1] - E[0]
    exact = expm(-t_ref * H)

    # Gate: positivity / reflection positivity
    tau = 0.05
    T = expm(-tau * V / 2) @ expm(-tau * 0.5 * g ** 2 * K) @ expm(-tau * V / 2)
    T = 0.5 * (T + T.T)
    rep.check("Gate: T_tau is positive semidefinite", np.linalg.eigvalsh(T).min() > -1e-12)
    ok = True
    Phi = rng.normal(size=(T.shape[0], 6))
    for j in range(1, 6):
        G = Phi.T @ np.linalg.matrix_power(T, j) @ Phi
        ok &= np.linalg.eigvalsh(0.5 * (G + G.T)).min() > -1e-10
    rep.check("Gate: reflection Gram matrices <phi_a, T^j phi_b> are PSD (j = 1..5)", ok)

    # (i) norm convergence and its order
    errs = []
    for n in (4, 8, 16, 32, 64):
        tau = t_ref / n
        T = expm(-tau * V / 2) @ expm(-tau * 0.5 * g ** 2 * K) @ expm(-tau * V / 2)
        errs.append(np.linalg.norm(np.linalg.matrix_power(T, n) - exact, 2))
    ratios = [errs[k] / errs[k + 1] for k in range(len(errs) - 1)]
    rep.check("(i) ||T_tau^{t/tau} - e^{-tH}|| -> 0, second order (ratio ~ 4 per halving)",
              all(3.5 < r < 4.5 for r in ratios) and errs[-1] < 1e-3,
              "errors " + ", ".join(f"{e:.1e}" for e in errs))

    # (i) ground projection and correlations
    tau = t_ref / 64
    T = expm(-tau * V / 2) @ expm(-tau * 0.5 * g ** 2 * K) @ expm(-tau * V / 2)
    lam, W = eigh(0.5 * (T + T.T))
    om_tau = W[:, -1]
    overlap = abs(om_tau @ Om)
    rep.check("(i) top eigenvector of T_tau -> ground state of H", overlap > 1 - 1e-6, f"|<.,.>| = {overlap:.9f}")
    F, Gm = obs["cos th1"], obs["cos th1"] + 0.5 * obs["cos(th1-th2)"]
    t = 0.75
    n = int(round(t / tau))
    lhs = np.vdot(om_tau, F @ np.linalg.matrix_power(T, n) @ Gm @ om_tau) / lam[-1] ** n
    rhs = np.vdot(Om, F @ expm(-t * (H - E[0] * np.eye(len(E)))) @ Gm @ Om)
    rep.check("(i) lim <F_0 G_t>_tau = <Omega, F e^{-t(H-E0)} G Omega>", abs(lhs - rhs) < 1e-3,
              f"{lhs.real:.6f} vs {rhs.real:.6f}")

    # (ii) decay rates
    rates = {}
    for name, X in obs.items():
        amps = np.abs(Uv.conj().T @ (X @ Om)) ** 2
        amps[0] = 0.0
        def C(tt):
            return float(np.sum(amps[1:] * np.exp(-tt * (E[1:] - E[0]))))
        if C(0.0) < 1e-14:
            continue
        t1, t2 = 40.0, 60.0
        rates[name] = -(math.log(C(t2)) - math.log(C(t1))) / (t2 - t1)
    combo = sum(rng.normal() * X for X in obs.values())
    amps = np.abs(Uv.conj().T @ (combo @ Om)) ** 2; amps[0] = 0
    Cc = lambda tt: float(np.sum(amps[1:] * np.exp(-tt * (E[1:] - E[0]))))
    rates["random combination"] = -(math.log(Cc(60.0)) - math.log(Cc(40.0))) / 20.0
    rep.check("(ii) every truncated correlation decays at rate >= delta",
              all(r >= delta - 1e-6 for r in rates.values()),
              "delta = %.5f; rates " % delta + ", ".join(f"{k}: {v:.5f}" for k, v in rates.items()))
    rep.check("(ii) inf over observables of the decay rate equals delta",
              abs(min(rates.values()) - delta) < 1e-3, f"min rate {min(rates.values()):.5f}")
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
