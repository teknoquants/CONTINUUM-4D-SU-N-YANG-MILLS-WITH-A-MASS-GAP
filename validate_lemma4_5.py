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
"""Validation of Lemmas 4 and 5 (matching across scales; gap scales with spacing).

Lemmas 4 and 5 are implications from open renormalization-group hypotheses. This script checks that
the implications are correct: it generates many synthetic RG data sets that satisfy the hypotheses
R, R', Y, Y', AF, AF+- and G4, G5 exactly, and verifies the conclusions with the constants stated in
the paper. It also checks the constants b0, b1 and the remark that motivates the two-loop scale.

What is checked
  - b1 = (34/3) N^2 / (16 pi^2)^2 = 17 N^2 / (384 pi^4).
  - Two-loop remark: ln(a1(g0)/a1(g)) - ln(a2(g0)/a2(g)) grows without bound as g -> 0 (so a bounded
    kappa fails against the one-loop scale), while relative to a2 the stopping scale k* defined by AF+-
    stays within kappa.
  - Lemma 4: delta_L(g) >= C4 a(g), C4 = theta (g0^2 c_N / 4) b^{-kappa} / a(g0).
  - Lemma 5 (i): delta_L(g) > 0 for all g > 0 (with Lemma 1 for g >= g0).
  - Lemma 5 (ii): C4 <= delta_L(g)/a(g) <= C5, C5 = Theta-bar D b^kappa / a(g0).
  - Descent algebra: iterating R gives delta^(0) >= b^{-k*} prod(1 - eta_k) delta^(k*).

What is NOT checked: that any actual renormalization-group map satisfies these hypotheses. That is
the open crux of the programme.
"""
import math
import sys

import numpy as np

from common import Report, c_N, b0, b1, a_two_loop, a_one_loop, g0 as g0_of


def main():
    rep = Report("Lemmas 4 and 5")
    rng = np.random.default_rng(5)

    ok = all(abs(b1(N) - (34 / 3) * N ** 2 / (16 * math.pi ** 2) ** 2) < 1e-15 for N in range(2, 8))
    rep.check("b1 = (34/3) N^2/(16 pi^2)^2 = 17 N^2/(384 pi^4)", ok)

    N, b, kappa = 3, 2, 1.0
    c1, c2 = 0.2, 1.5
    G0 = g0_of(N, c1, c2)
    endpoint_lo = G0 ** 2 * c_N(N) / 4
    D = 5 * endpoint_lo

    # two-loop remark
    gs = [G0 * x for x in (0.5, 0.3, 0.2, 0.15, 0.12, 0.1)]
    diff = [math.log(a_one_loop(G0, N) / a_one_loop(g, N)) - math.log(a_two_loop(G0, N) / a_two_loop(g, N))
            for g in gs]
    rep.check("one-loop vs two-loop stopping scales differ by an unbounded log(1/g^2) term",
              all(diff[k + 1] > diff[k] for k in range(len(diff) - 1)) and
              abs(diff[-1] - (b1(N) / (2 * b0(N) ** 2)) * math.log(gs[0] ** 2 / gs[-1] ** 2) - diff[0]) < 1e-9,
              "differences " + ", ".join(f"{d:.3f}" for d in diff))

    # synthetic RG data satisfying all hypotheses
    S = 1.0                      # sum_k -ln(1 - eta_k) <= S  ==>  theta >= e^{-S}
    theta = math.exp(-S)
    Theta_bar = math.exp(S)      # prod (1 + eta_k) <= e^{sum eta_k} <= e^{S}
    C4 = theta * (G0 ** 2 * c_N(N) / 4) * b ** (-kappa) / a_two_loop(G0, N)
    C5 = Theta_bar * D * b ** kappa / a_two_loop(G0, N)

    ok4 = ok5 = okdesc = okpos = True
    worst_lo, worst_hi = float("inf"), 0.0
    for _ in range(4000):
        g = G0 * rng.uniform(0.25, 0.999)
        target = math.log(a_two_loop(G0, N) / a_two_loop(g, N)) / math.log(b)
        kstar = max(1, int(round(target + rng.uniform(-kappa, kappa))))
        kstar = min(max(kstar, math.ceil(target - kappa)), math.floor(target + kappa))
        if kstar < 1:
            continue
        # AF+- holds by construction: |kstar - target| <= kappa
        w = rng.dirichlet(np.ones(kstar)) * S * rng.uniform(0, 1)
        eta = 1 - np.exp(-w)                       # sum -ln(1-eta) <= S, and sum eta <= S
        delta = rng.uniform(endpoint_lo, D)        # Y and Y'
        delta_k = delta
        lower_chain = upper_chain = delta
        for k in range(kstar, 0, -1):
            x = rng.uniform(1 - eta[k - 1], 1 + eta[k - 1])   # R and R'
            delta_k = delta_k * x / b
            lower_chain *= (1 - eta[k - 1]) / b
            upper_chain *= (1 + eta[k - 1]) / b
        d0 = delta_k
        okdesc &= lower_chain * (1 - 1e-12) <= d0 <= upper_chain * (1 + 1e-12)
        a = a_two_loop(g, N)
        ok4 &= d0 >= C4 * a * (1 - 1e-12)
        ok5 &= C4 * (1 - 1e-12) <= d0 / a <= C5 * (1 + 1e-12)
        okpos &= d0 > 0
        worst_lo, worst_hi = min(worst_lo, d0 / a / C4), max(worst_hi, d0 / a / C5)
    rep.check("descent: b^{-k*} prod(1-eta) delta^(k*) <= delta^(0) <= b^{-k*} prod(1+eta) delta^(k*)", okdesc)
    rep.check("Lemma 4: delta_L(g) >= C4 a(g)", ok4, f"min (delta/a)/C4 = {worst_lo:.3f}")
    rep.check("Lemma 5 (ii): C4 <= delta_L(g)/a(g) <= C5", ok5, f"max (delta/a)/C5 = {worst_hi:.3f}")
    okpos &= all(g * g * c_N(N) / 4 > 0 for g in np.linspace(G0, 5 * G0, 20))
    rep.check("Lemma 5 (i): delta_L(g) > 0 for all g > 0 (Lemma 1 for g >= g0)", okpos)

    # hypotheses are needed: violating G4 (theta -> 0) breaks the lower bound
    g = 0.3 * G0
    target = math.log(a_two_loop(G0, N) / a_two_loop(g, N)) / math.log(b)
    kstar = math.ceil(target)
    d0 = endpoint_lo * b ** (-kstar) * (1 - 0.5) ** kstar   # eta_k = 1/2 at every step: theta -> 0
    rep.check("sanity: without G4 (eta_k = 1/2 at every scale) the bound C4 a(g) fails",
              d0 < C4 * a_two_loop(g, N), f"delta/a = {d0 / a_two_loop(g, N):.3e} < C4 = {C4:.3e}")
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
