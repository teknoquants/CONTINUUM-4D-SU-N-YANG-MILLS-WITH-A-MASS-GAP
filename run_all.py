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
"""Run every validation script for "Continuum 4D SU(N) Yang-Mills with a Mass Gap".

Usage:  python run_all.py
Exit code 0 if every check passes.
"""
import sys
import time

import validate_lemma1
import validate_lemma2
import validate_lemma3
import validate_lemma4_5
import validate_theorem

SUITES = [
    ("Lemma 1 - strong coupling", validate_lemma1),
    ("Lemma 2 - Hamiltonian-Euclidean bridge", validate_lemma2),
    ("Lemma 3 - exact 2D matching", validate_lemma3),
    ("Lemmas 4 and 5 - matching and gap scaling (implications)", validate_lemma4_5),
    ("Theorem - conditional continuum reconstruction (logical steps)", validate_theorem),
]


def main():
    results = []
    for title, mod in SUITES:
        print(f"\n=== {title} ===")
        t = time.time()
        ok = mod.main()
        results.append((title, ok, time.time() - t))
    print("\n=== Summary ===")
    for title, ok, dt in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {title}  ({dt:.1f} s)")
    print("\nNot validated (open hypotheses of the paper): certified Yarotsky constants and the link-embedding\n"
          "estimate (G1), the expansion E(g,K), the RG hypotheses R, R', Y, Y', AF, AF+-, G4, G5, and the\n"
          "continuum hypotheses C1-C4. The checks above validate proved statements, the algebra of every\n"
          "implication, and finite-dimensional versions of the operator arguments.")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
