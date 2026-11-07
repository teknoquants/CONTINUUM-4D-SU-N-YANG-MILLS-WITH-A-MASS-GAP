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
"""Validation of Lemma 1 (strong coupling).

What is checked
  (a) Threshold algebra: 6N/(g^4 c_N) <= beta*  <=>  g >= g0(N), and 1 - c2*beta* >= 1/2.
  (b) Plaquette adjacency graph of the cubic lattice has maximum degree 12.
  (c) Connected plaquette sets of size n containing a fixed plaquette number at most (12e)^n.
  (d) At most 8(r+s) plaquettes contain a link of an r x s rectangle.
  (e) Area count: every Z_N 2-chain with boundary C (an r x s rectangle) has >= rs plaquettes
      (exact integer programme, N = 2 and N = 3, inside a padded box).
  (f) The summation bound 16(r+s) q^{rs} <= M e^{-sigma rs}, sigma = (1/2) ln(1/q), M = 32/(e sigma).
  (g) Center selection rule for SU(2) (Monte Carlo): <W(C) chi(U_p1)> = 0 when {p1} carries no
      chain spanning C, while <W(C) chi(U_p1) chi(U_p2)> = 1/4 for the spanning pair.
  (h) Illustration of the Yarotsky mechanism used in part (i): a chain of three-level sites with
      on-site gap 1 and small bounded nearest-neighbour perturbation keeps a gap >= 1/2, uniformly in
      the chain length.

What is NOT checked: the certified constants c1, c2 of Yarotsky's theorem and the expansion
hypothesis E(g, K); both are open hypotheses of the paper.
"""
import itertools
import math
import sys

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from common import Report, c_N, beta_star, g0, haar_su2

E = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def add(x, v, s=1):
    return tuple(a + s * b for a, b in zip(x, v))


def box_geometry(nx, ny, nz):
    verts = [(x, y, z) for x in range(nx + 1) for y in range(ny + 1) for z in range(nz + 1)]
    inside = lambda v: 0 <= v[0] <= nx and 0 <= v[1] <= ny and 0 <= v[2] <= nz
    links = [(v, i) for v in verts for i in range(3) if inside(add(v, E[i]))]
    plaqs = [(v, i, j) for v in verts for i in range(3) for j in range(i + 1, 3)
             if inside(add(add(v, E[i]), E[j]))]
    return links, plaqs


def plaq_boundary(p):
    v, i, j = p
    return [((v, i), +1), ((add(v, E[i]), j), +1), ((add(v, E[j]), i), -1), ((v, j), -1)]


def adjacency(plaqs):
    by_link = {}
    for p in plaqs:
        for l, _ in plaq_boundary(p):
            by_link.setdefault(l, []).append(p)
    nb = {p: set() for p in plaqs}
    for ps in by_link.values():
        for a in ps:
            for b in ps:
                if a != b:
                    nb[a].add(b)
    return nb


def count_animals(nb, p0, nmax):
    level = {frozenset([p0])}
    counts = [1]
    for _ in range(2, nmax + 1):
        nxt = set()
        for S in level:
            frontier = set().union(*(nb[q] for q in S)) - S
            for q in frontier:
                nxt.add(S | {q})
        level = nxt
        counts.append(len(level))
    return counts


def min_chain_support(r, s, N):
    """Minimum support of a Z_N 2-chain with boundary the r x s rectangle, inside a padded box."""
    nx, ny, nz = r + 2, s + 2, 2
    links, plaqs = box_geometry(nx, ny, nz)
    li = {l: k for k, l in enumerate(links)}
    rect = [((1 + a, 1 + b, 1), 0, 1) for a in range(r) for b in range(s)]
    target = np.zeros(len(links), dtype=int)
    for p in rect:
        for l, o in plaq_boundary(p):
            target[li[l]] += o
    target %= N
    P, Lk = len(plaqs), len(links)
    # variables: x_p in {0..N-1}, y_p in {0,1}, k_l integer
    nv = 2 * P + Lk
    A_eq = np.zeros((Lk, nv))
    for pi, p in enumerate(plaqs):
        for l, o in plaq_boundary(p):
            A_eq[li[l], pi] += o
    for l in range(Lk):
        A_eq[l, 2 * P + l] = -N
    A_ub = np.zeros((P, nv))  # x_p - (N-1) y_p <= 0
    for pi in range(P):
        A_ub[pi, pi] = 1
        A_ub[pi, P + pi] = -(N - 1)
    c = np.concatenate([np.zeros(P), np.ones(P), np.zeros(Lk)])
    lb = np.concatenate([np.zeros(P), np.zeros(P), -8 * np.ones(Lk)])
    ub = np.concatenate([(N - 1) * np.ones(P), np.ones(P), 8 * np.ones(Lk)])
    res = milp(c, integrality=np.ones(nv),
               constraints=[LinearConstraint(A_eq, target, target), LinearConstraint(A_ub, -np.inf, 0)],
               bounds=Bounds(lb, ub))
    return int(round(res.fun)), res.status


def main():
    rep = Report("Lemma 1")
    rng = np.random.default_rng(2026)

    # (a) threshold algebra
    ok = True
    for N in (2, 3, 4):
        for c1, c2 in [(0.1, 2.0), (0.02, 0.5), (0.3, 7.0)]:
            bs, G0 = beta_star(c1, c2), g0(N, c1, c2)
            ok &= (1 - c2 * bs) >= 0.5 - 1e-12
            for g in np.linspace(0.5 * G0, 2 * G0, 61):
                lhs = 6 * N / (g ** 4 * c_N(N)) <= bs * (1 + 1e-12)
                ok &= lhs == (g >= G0 * (1 - 1e-12))
    rep.check("(a) 6N/(g^4 c_N) <= beta*  iff  g >= g0(N);  1 - c2 beta* >= 1/2", ok)

    # (b), (c) plaquette graph
    links, plaqs = box_geometry(6, 6, 6)
    nb = adjacency(plaqs)
    maxdeg = max(len(v) for v in nb.values())
    rep.check("(b) plaquette adjacency degree <= 12", maxdeg == 12, f"max degree {maxdeg}")
    p0 = ((3, 3, 3), 0, 1)
    counts = count_animals(nb, p0, 4)
    bound_ok = all(cn <= (12 * math.e) ** n for n, cn in enumerate(counts, start=1))
    rep.check("(c) connected plaquette sets of size n through p0 <= (12e)^n", bound_ok,
              "counts n=1..4: " + ", ".join(map(str, counts)))

    # (d) plaquettes containing a link of an r x s rectangle
    ok = True
    for r, s in [(1, 1), (2, 3), (3, 3), (4, 2)]:
        rect_links = set()
        for a in range(r):
            for b in range(s):
                for l, _ in plaq_boundary(((1 + a, 1 + b, 3), 0, 1)):
                    rect_links.add(l)
        # boundary links of the rectangle = links in exactly one rectangle plaquette
        mult = {}
        for a in range(r):
            for b in range(s):
                for l, _ in plaq_boundary(((1 + a, 1 + b, 3), 0, 1)):
                    mult[l] = mult.get(l, 0) + 1
        Clinks = {l for l, m in mult.items() if m == 1}
        touching = {p for p in plaqs if any(l in Clinks for l, _ in plaq_boundary(p))}
        ok &= len(touching) <= 8 * (r + s) and len(Clinks) == 2 * (r + s)
    rep.check("(d) #plaquettes containing a link of C <= 8(r+s)", ok)

    # (e) area count by exact integer programming
    ok, det = True, []
    for N in (2, 3):
        for r, s in [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]:
            m, status = min_chain_support(r, s, N)
            ok &= (status == 0) and (m == r * s)
            det.append(f"N={N},{r}x{s}:{m}")
    rep.check("(e) min support of Z_N 2-chain with boundary C equals rs", ok, "; ".join(det))

    # (f) summation bound
    ok = True
    for q in np.linspace(0.01, 0.5, 50):
        sig = 0.5 * math.log(1 / q)
        M = 32 / (math.e * sig)
        for r in range(1, 30):
            for s in range(1, 30):
                ok &= 16 * (r + s) * q ** (r * s) <= M * math.exp(-sig * r * s) * (1 + 1e-12)
    rep.check("(f) 16(r+s) q^{rs} <= M e^{-sigma rs} on a grid", ok)

    # (g) center selection rule, SU(2) Monte Carlo
    n = 400_000
    h00, h10, h01, h11, v00, v10, v20 = (haar_su2(n, rng) for _ in range(7))
    dag = lambda X: np.conj(np.transpose(X, (0, 2, 1)))
    Up1 = h00 @ v10 @ dag(h01) @ dag(v00)
    Up2 = h10 @ v20 @ dag(h11) @ dag(v10)
    UC = h00 @ h10 @ v20 @ dag(h11) @ dag(h01) @ dag(v00)
    tr = lambda X: np.real(np.trace(X, axis1=1, axis2=2))
    W = tr(UC) / 2
    m0 = np.mean(W * tr(Up1)); s0 = np.std(W * tr(Up1)) / math.sqrt(n)
    m1 = np.mean(W * tr(Up1) * tr(Up2)); s1 = np.std(W * tr(Up1) * tr(Up2)) / math.sqrt(n)
    rep.check("(g) <W(C) chi(U_p1)> = 0  (no spanning chain on {p1})", abs(m0) < 5 * s0 + 1e-3,
              f"{m0:.4f} +/- {s0:.4f}")
    rep.check("(g) <W(C) chi(U_p1) chi(U_p2)> = 1/4  (spanning pair)", abs(m1 - 0.25) < 5 * s1 + 1e-3,
              f"{m1:.4f} +/- {s1:.4f}")

    # (h) Yarotsky-type illustration
    h = np.diag([0.0, 1.0, 1.5])
    beta = 0.05
    gaps = []
    for L in range(2, 7):
        d = 3 ** L
        H = np.zeros((d, d))
        for k in range(L):
            H += np.kron(np.kron(np.eye(3 ** k), h), np.eye(3 ** (L - k - 1)))
        for k in range(L - 1):
            X = rng.normal(size=(9, 9)); X = X + X.T
            X *= beta / np.linalg.norm(X, 2)
            H += np.kron(np.kron(np.eye(3 ** k), X), np.eye(3 ** (L - k - 2)))
        ev = np.linalg.eigvalsh(H)
        gaps.append(ev[1] - ev[0])
    rep.check("(h) illustration: gap >= 1/2 uniformly in chain length (beta = 0.05)",
              min(gaps) >= 0.5 and max(gaps) - min(gaps) < 0.2,
              "gaps " + ", ".join(f"{x:.3f}" for x in gaps))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
