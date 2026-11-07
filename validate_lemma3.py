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
"""Validation of Lemma 3 (exact two-dimensional matching) for G = SU(2).

What is checked
  - Heat kernel: K_t is a probability density with respect to Haar measure, and
    int K_t chi_n dU = n e^{-t C_n / 2} (character eigenvalues), by quadrature.
  - Gluing / convolution K_s * K_t = K_{s+t}, by Monte Carlo on products of independent
    conjugation-invariant samples (the step used at every gluing in the proof of (i)).
  - (i)  Blocking: a coarse plaquette built from b^2 fine plaquettes carries K_{b^2 t}; character
         expectations of products of b^2 samples match those of K_{b^2 t} (b = 2, 3).
  - (ii) Area law: <W(C)> = exp(-t c_N |A(C)|/2) for |A(C)| independent, randomly conjugated
         plaquette lassos (c_N = 3/2 for SU(2)).
  - (iii) The step count k = max{0, ceil(ln(t0/t)/ln(b^2))} gives b^{2k} t >= t0 and is minimal.

What is NOT checked: the three-dimensional and 2+1 warm-up, which the paper leaves open.
"""
import math
import sys

import numpy as np

from common import (Report, c_N, haar_su2, heat_kernel_su2, sample_heat_kernel_su2, chi_su2,
                    su2_casimir)


def main():
    rep = Report("Lemma 3")
    rng = np.random.default_rng(11)
    th = np.linspace(1e-6, math.pi - 1e-6, 40001)
    haar_w = (2 / math.pi) * np.sin(th) ** 2

    # heat kernel normalisation and character eigenvalues
    ok, det = True, []
    for t in (0.2, 0.7, 1.5):
        K = heat_kernel_su2(th, t)
        norm = np.trapezoid(K * haar_w, th)
        ok &= abs(norm - 1) < 1e-6 and K.min() > -1e-9
        for n in (2, 3, 4):
            val = np.trapezoid(K * np.sin(n * th) / np.sin(th) * haar_w, th)
            ok &= abs(val - n * math.exp(-t * su2_casimir(n) / 2)) < 1e-6
        det.append(f"t={t}: norm {norm:.8f}")
    rep.check("heat kernel: probability density, int K_t chi_n = n e^{-t C_n/2}", ok, "; ".join(det))

    nsamp = 300_000

    # gluing: K_s * K_t = K_{s+t}
    s, t = 0.3, 0.5
    P = sample_heat_kernel_su2(nsamp, s, rng) @ sample_heat_kernel_su2(nsamp, t, rng)
    ok, det = True, []
    for n in (2, 3):
        m = np.mean(chi_su2(P, n)); se = np.std(chi_su2(P, n)) / math.sqrt(nsamp)
        target = n * math.exp(-(s + t) * su2_casimir(n) / 2)
        ok &= abs(m - target) < 5 * se + 2e-3
        det.append(f"n={n}: {m:.4f} vs {target:.4f}")
    rep.check("gluing: K_s * K_t = K_{s+t} (character test)", ok, "; ".join(det))

    # (i) blocking b = 2, 3
    t = 0.15
    ok, det = True, []
    for b in (2, 3):
        prod = np.broadcast_to(np.eye(2, dtype=complex), (nsamp, 2, 2)).copy()
        for _ in range(b * b):
            prod = prod @ sample_heat_kernel_su2(nsamp, t, rng)
        for n in (2, 3):
            m = np.mean(chi_su2(prod, n)); se = np.std(chi_su2(prod, n)) / math.sqrt(nsamp)
            target = n * math.exp(-b * b * t * su2_casimir(n) / 2)
            ok &= abs(m - target) < 5 * se + 2e-3
            det.append(f"b={b},n={n}: {m:.4f} vs {target:.4f}")
    rep.check("(i) blocking: b^2 fine plaquettes carry K_{b^2 t}", ok, "; ".join(det))

    # (ii) area law with random conjugations (lassos conjugated by tree paths)
    t = 0.2
    ok, det = True, []
    for A in (1, 2, 4, 6):
        hol = np.broadcast_to(np.eye(2, dtype=complex), (nsamp, 2, 2)).copy()
        for _ in range(A):
            Vc = haar_su2(nsamp, rng)
            L = sample_heat_kernel_su2(nsamp, t, rng)
            hol = hol @ (Vc @ L @ np.conj(np.transpose(Vc, (0, 2, 1))))
        W = np.real(np.trace(hol, axis1=1, axis2=2)) / 2
        m, se = np.mean(W), np.std(W) / math.sqrt(nsamp)
        target = math.exp(-t * c_N(2) * A / 2)
        ok &= abs(m - target) < 5 * se + 2e-3
        det.append(f"A={A}: {m:.4f} vs {target:.4f}")
    rep.check("(ii) <W(C)> = exp(-t c_N |A(C)| / 2)", ok, "; ".join(det))

    # (iii) step count
    ok = True
    for b in (2, 3, 4):
        for t in np.geomspace(1e-4, 10, 40):
            for t0 in (0.5, 1.0, 5.0):
                k = max(0, math.ceil(math.log(t0 / t) / math.log(b * b)))
                ok &= b ** (2 * k) * t >= t0 * (1 - 1e-12)
                if k > 0:
                    ok &= b ** (2 * (k - 1)) * t < t0
    rep.check("(iii) k = max{0, ceil(ln(t0/t)/ln b^2)} reaches t0 and is minimal", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
