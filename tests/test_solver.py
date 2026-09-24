"""Brute-force differential tests for the exact solver.

Every random small instance is solved independently by full enumeration of
the (small) counting box; the branch-and-bound solver must agree on the
nearest reachable masses, the minimum particle count, the number of optimal
vectors and the canonical witness set.
"""

from __future__ import annotations

import itertools
import random
import unittest

from app.solver import Component, Solver


def brute_force(masses, los, his, T, tol):
    reachable = {}
    ranges = [range(lo, hi + 1) for lo, hi in zip(los, his)]
    for vec in itertools.product(*ranges):
        mass = sum(c * m for c, m in zip(vec, masses))
        reachable.setdefault(mass, []).append(vec)
    if not reachable:
        return None
    best_dist = min(abs(m - T) for m in reachable)
    if best_dist > tol:
        below = [m for m in reachable if m <= T]
        above = [m for m in reachable if m >= T]
        return {
            "status": "unsatisfiable",
            "best_dist": best_dist,
            "below": max(below) if below else None,
            "above": min(above) if above else None,
        }
    opt_masses = [m for m in reachable if abs(m - T) == best_dist]
    winners = []
    best_particles = None
    for m in opt_masses:
        for vec in reachable[m]:
            pc = sum(vec)
            if best_particles is None or pc < best_particles:
                best_particles = pc
                winners = [vec]
            elif pc == best_particles:
                winners.append(vec)
    return {
        "status": "optimal",
        "best_dist": best_dist,
        "opt_masses": sorted(opt_masses),
        "particles": best_particles,
        "num": len(winners),
        "winners": sorted(winners),
    }


class SolverTests(unittest.TestCase):
    def _run_case(self, masses, los, his, T, tol):
        comps = [Component(chr(ord("A") + i), masses[i], los[i], his[i])
                 for i in range(len(masses))]
        solver = Solver(comps, T, tol, node_budget=1_000_000)
        result = solver.solve(max_collect=10_000)
        expected = brute_force(masses, los, his, T, tol)

        below = result["nearest_below"]
        above = result["nearest_above"]
        eb = max((m for m in self._all_masses(masses, los, his) if m <= T),
                 default=None)
        ea = min((m for m in self._all_masses(masses, los, his) if m >= T),
                 default=None)
        self._check_mass(below, eb, T)
        self._check_mass(above, ea, T)

        if expected["status"] == "unsatisfiable":
            self.assertEqual(result["status"], "unsatisfiable")
            self.assertEqual(result["best_distance"], expected["best_dist"])
            return

        self.assertEqual(result["status"], "optimal")
        self.assertEqual(result["best_distance"], expected["best_dist"])
        self.assertEqual(result["optimal_masses"], expected["opt_masses"])
        self.assertEqual(result["particle_count"], expected["particles"])
        self.assertEqual(result["num_optimal_explanations"], expected["num"])
        self.assertEqual(result["unique"], expected["num"] == 1)
        self.assertEqual(result["vectors"],
                         [tuple(v) for v in expected["winners"]])
        # All reported masses reconcile exactly with their vectors.
        for vec in result["vectors"]:
            self.assertEqual(
                sum(vec[i] * solver.m[i] for i in range(len(masses))),
                next(m for m in expected["opt_masses"]
                     if vec in [tuple(x) for x in
                                self._vectors_at(masses, los, his, m)]))

    @staticmethod
    def _all_masses(masses, los, his):
        ranges = [range(lo, hi + 1) for lo, hi in zip(los, his)]
        return {sum(c * m for c, m in zip(vec, masses))
                for vec in itertools.product(*ranges)}

    @staticmethod
    def _vectors_at(masses, los, his, mass):
        ranges = [range(lo, hi + 1) for lo, hi in zip(los, his)]
        return [vec for vec in itertools.product(*ranges)
                if sum(c * m for c, m in zip(vec, masses)) == mass]

    @staticmethod
    def _check_mass(witness, expected_mass, T):
        if expected_mass is None:
            assert witness is None
        else:
            assert witness is not None
            assert witness["total_mass"] == expected_mass

    def test_random_instances(self):
        rng = random.Random(20260919)
        for trial in range(400):
            k = rng.randint(2, 5)
            masses = [rng.randint(1, 40) for _ in range(k)]
            los = [rng.randint(0, 2) for _ in range(k)]
            his = [lo + rng.randint(0, 5) for lo in los]
            T = rng.randint(0, sum(m * h for m, h in zip(masses, his)) + 5)
            tol = rng.choice([0, 1, 2, 5, rng.randint(0, 20)])
            with self.subTest(trial=trial, masses=masses, los=los,
                              his=his, T=T, tol=tol):
                self._run_case(masses, los, his, T, tol)

    def test_exact_target_multiple_optimal_masses(self):
        # masses 6 and 10: target 30 -> below 30 (5*6 and 3*10, etc.) exact.
        # Also a crafted symmetric case: masses 4 and 6, T = 12, tol 0.
        self._run_case([4, 6], [0, 0], [5, 5], 12, 0)

    def test_tie_on_both_sides(self):
        # T = 11 with masses 4 and 6: reachable 8, 10, 12, ... -> distance 1
        # both sides (10 below, 12 above); tol = 1 includes both.
        self._run_case([4, 6], [0, 0], [6, 6], 11, 1)

    def test_unsatisfiable_with_tight_tolerance(self):
        self._run_case([7, 13], [1, 1], [4, 4], 50, 1)

    def test_nonzero_lower_bounds(self):
        self._run_case([3, 5], [2, 1], [6, 5], 25, 0)
        self._run_case([3, 5], [2, 1], [6, 5], 25, 2)

    def test_target_below_box(self):
        # Everything is above T; nearest-below must be None.
        comps = [Component("A", 10, 2, 5), Component("B", 20, 1, 3)]
        res = Solver(comps, 5, 2, node_budget=100_000).solve(100)
        self.assertEqual(res["status"], "unsatisfiable")
        self.assertIsNone(res["nearest_below"])
        self.assertEqual(res["nearest_above"]["total_mass"], 40)

    def test_target_above_box(self):
        comps = [Component("A", 10, 2, 5), Component("B", 20, 1, 3)]
        res = Solver(comps, 10_000, 2, node_budget=100_000).solve(100)
        self.assertEqual(res["status"], "unsatisfiable")
        self.assertIsNone(res["nearest_above"])
        self.assertEqual(res["nearest_below"]["total_mass"], 110)

    def test_canonical_ordering_by_id(self):
        comps = [Component("z", 4, 0, 6), Component("a", 6, 0, 6)]
        res = Solver(comps, 11, 1).solve(100)
        self.assertEqual(res["component_order"], ["a", "z"])
        for vec in res["vectors"]:
            self.assertEqual(len(vec), 2)

    def test_large_bounds_not_expanded(self):
        # Bounds up to 1,000,000 must answer instantly without enumeration.
        comps = [Component("A", 123_456_789, 0, 1_000_000),
                 Component("B", 987_654_321, 0, 1_000_000),
                 Component("C", 555_555_557, 0, 1_000_000)]
        T = 12_345_678_901_234
        res = Solver(comps, T, 1_000_000, node_budget=10_000_000).solve(10)
        self.assertEqual(res["status"], "optimal")
        # direct recompute via order mapping
        order = res["component_order"]
        mm = {"A": 123_456_789, "B": 987_654_321, "C": 555_555_557}
        for vec in res["vectors"]:
            mass = sum(vec[i] * mm[order[i]] for i in range(3))
            self.assertEqual(abs(mass - T), res["best_distance"])
            self.assertLessEqual(abs(mass - T), 1_000_000)


if __name__ == "__main__":
    unittest.main()
