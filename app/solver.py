"""Exact branch-and-bound inversion over a bounded counting box.

Model (all quantities are arbitrary-precision Python ``int``)::

    M = sum_i x_i * m_i,        lo_i <= x_i <= hi_i,   m_i > 0

The service must never *expand* a counting interval.  Everything here walks the
box with recursive branch-and-bound; no interval is materialized, no dynamic
program is laid out over any count interval, and no general optimization
solver is used.

Two-level optimization
----------------------
1. **Quality**  minimize |M - target| over all reachable masses in the box.
   Exact nearest reachable masses on either side of the target are found with a
   DFS pruned by suffix mass windows, suffix-gcd congruence feasibility and an
   incumbent seeded by greedy filling/trimming.
2. **Particle count**  among *every* count vector attaining an optimal mass,
   minimize sum x_i.  A memoized suffix recursion over ``(position, residual)``
   computes the exact minimum and the exact (arbitrary precision) number of
   attaining vectors, which decides uniqueness; concrete witnesses are
   enumerated on demand in lexicographically canonical order.

Both phases only ever use integer arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Optional

# The unlimited-coin residue oracle costs O(n * m_min) time and memory; only
# build it when the smallest mass is at most this many micro-daltons.
ORACLE_MAX_MODULUS = 2_000_000
# Gates scan residues directly while the incumbent gap window is this small;
# wider windows fall back to block-indexed range minima.
ORACLE_SCAN_LIMIT = 512
_ORACLE_BLOCK = 256
_INF = 10**30


class BudgetExceeded(Exception):
    """Raised when a search visits more nodes than the configured budget."""


@dataclass(frozen=True)
class Component:
    id: str
    mass: int
    lo: int
    hi: int


_INFEASIBLE = object()


def _ceil_div(a: int, b: int) -> int:
    """ceil(a / b) for b > 0, exact for possibly-negative a."""
    return -((-a) // b)


class Solver:
    def __init__(self, components: list[Component], target: int, tolerance: int,
                 node_budget: int = 5_000_000):
        # Canonical coordinate order: by component identifier.
        comps = sorted(components, key=lambda c: c.id)
        self.ids = [c.id for c in comps]
        self.m = [c.mass for c in comps]
        self.lo = [c.lo for c in comps]
        self.hi = [c.hi for c in comps]
        self.n = len(comps)
        self.T = target
        self.tol = tolerance
        self.node_budget = node_budget
        self.nodes = 0

        self.base = sum(self.lo[i] * self.m[i] for i in range(self.n))
        self.top = sum(self.hi[i] * self.m[i] for i in range(self.n))
        self.cap = [self.hi[i] - self.lo[i] for i in range(self.n)]

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes > self.node_budget:
            raise BudgetExceeded(f"search visited {self.nodes} nodes")

    # ------------------------------------------------------------------ #
    # Phase 1: nearest reachable mass on each side of the target
    # ------------------------------------------------------------------ #

    def _greedy_below(self) -> Optional[tuple[int, tuple[int, ...]]]:
        """Largest-ish reachable mass <= T via greedy filling (seed only)."""
        if self.base > self.T:
            return None
        x = self.lo[:]
        total = self.base
        for j in sorted(range(self.n), key=lambda i: -self.m[i]):
            inc = min(self.cap[j], (self.T - total) // self.m[j])
            if inc:
                x[j] += inc
                total += inc * self.m[j]
        return total, tuple(x)

    def _greedy_above(self) -> Optional[tuple[int, tuple[int, ...]]]:
        """Smallest-ish reachable mass >= T via greedy trimming (seed only)."""
        if self.top < self.T:
            return None
        x = self.hi[:]
        total = self.top
        for j in sorted(range(self.n), key=lambda i: -self.m[i]):
            dec = min(x[j] - self.lo[j], (total - self.T) // self.m[j])
            if dec:
                x[j] -= dec
                total -= dec * self.m[j]
        return total, tuple(x)

    # ------------------------------------------------------------------ #
    # Phase 1: nearest reachable mass on each side of the target
    # ------------------------------------------------------------------ #

    def _build_oracle(self):
        """Residue shortest-path oracle over *unbounded* coins.

        Chooses the smallest mass m0 as modulus and computes, for every
        residue r, the minimum representable mass ``o[r]`` congruent to
        r (mod m0) over an *unbounded* relaxation of every coin (count caps
        ignored).  Because m0 itself is a coin, the unbounded representable
        values of residue r are exactly ``o[r] + z*m0`` for z >= 0.

        Per added coin the update is a min-plus closure on residue cycles;
        each cycle is covered by one forward and one backward sweep, giving
        O(m0) work per coin -- independent of all count bounds, so no
        counting interval is ever expanded.

        Returns ``(m0_index, m0, dist)`` or ``None`` when m0 is too large.
        """
        k = min(range(self.n), key=lambda i: self.m[i])
        m0 = self.m[k]
        if m0 > ORACLE_MAX_MODULUS:
            return None
        dist = [_INF] * m0
        dist[0] = 0

        for w in self.m:
            r = w % m0
            if r == 0:
                # Residue-0 coin: only useful at value 0 (other multiples are
                # strictly heavier residue-0 sums).
                continue
            g = gcd(r, m0)
            length = m0 // g
            for s in range(g):
                # Cycle v_t = (s + t*r) mod m0, t = 0..length-1.
                orig = [0] * length
                v = s
                for t in range(length):
                    orig[t] = dist[v]
                    v = (v + r) % m0
                wr = t  # silence linters; recomputed below implicitly
                del wr
                # Non-wrapping predecessors j <= t:
                #   best[t] = min_j (orig[j] + (t-j)*w)
                # Wrapping predecessors j > t:
                #   best[t] = min_j (orig[j] + (t + length - j)*w)
                best = [_INF] * length
                run = _INF
                for t in range(length):
                    cand = orig[t] - t * w
                    if cand < run:
                        run = cand
                    bt = run + t * w
                    if bt < best[t]:
                        best[t] = bt
                suf = _INF
                for t in range(length - 1, -1, -1):
                    bt = suf + (t + length) * w
                    if bt < best[t]:
                        best[t] = bt
                    cand = orig[t] - t * w
                    if cand < suf:
                        suf = cand
                v = s
                for t in range(length):
                    if best[t] < dist[v]:
                        dist[v] = best[t]
                    v = (v + r) % m0

        if dist[0] != 0:
            dist[0] = 0
        return k, m0, dist

    def _extreme(self, side: int, oracle=None) -> Optional[tuple[int, tuple[int, ...]]]:
        """Extreme reachable mass relative to the target.

        side = -1  ->  maximize M with M <= T (nearest reachable mass below)
        side = +1  ->  minimize M with M >= T (nearest reachable mass above)

        Returns ``(mass, count_vector)`` or ``None`` when no reachable mass
        exists on that side of the target.
        """
        n, m, T = self.n, self.m, self.T

        # Suffix information over *effective* counts y_i = x_i - lo_i.
        smax = [0] * (n + 1)
        sg = [0] * (n + 2)
        for i in range(n - 1, -1, -1):
            smax[i] = smax[i + 1] + self.cap[i] * m[i]
            sg[i] = gcd(m[i], sg[i + 1])

        seed = self._greedy_below() if side == -1 else self._greedy_above()
        best: Optional[int] = seed[0] if seed is not None else None
        best_vec: Optional[tuple[int, ...]] = seed[1] if seed is not None else None
        if best == T:
            return best, best_vec  # type: ignore[return-value]

        om: int = oracle[1] if oracle else 0
        od: Optional[list[int]] = oracle[2] if oracle else None
        x = self.lo[:]

        def oracle_gate(lo_q: int, hi_q: int) -> bool:
            """O(1) exact necessary condition over the unbounded relaxation.

            Rejects only when no value of the required residue class can fall
            inside [lo_q, hi_q] even with caps removed -- hence rejection is
            sound for the true bounded suffix.
            """
            if od is None:
                return True
            d = od[lo_q % om]
            if d == _INF or d > hi_q:
                return False
            # Smallest representable value >= lo_q in this residue class.
            if d < lo_q:
                d += _ceil_div(lo_q - d, om) * om
            return d <= hi_q

        def suffix_gate(i: int, S: int) -> bool:
            """Prune node (i, S): suffix range, gcd congruence, incumbent."""
            nonlocal best, best_vec
            g = sg[i]
            if side == -1:
                hi_q = min(smax[i], T - S)
                if hi_q < 0:
                    return False  # S > T and all masses are positive
                q = hi_q if g == 0 else hi_q - (hi_q % g)
                if q < 0:
                    return False
                if best is not None and S + q <= best:
                    return False
                lo_q = max(0, best + 1 - S) if best is not None else 0
                if lo_q > hi_q:
                    return False
                return oracle_gate(lo_q, hi_q)
            lo_q = max(0, T - S)
            if lo_q > smax[i]:
                return False  # even a full suffix cannot reach T
            if g == 0:
                q = 0
            else:
                q = lo_q + ((-lo_q) % g)
            if q > smax[i]:
                return False
            if best is not None and S + q >= best:
                return False
            hi_q = min(smax[i], best - 1 - S) if best is not None else smax[i]
            if lo_q > hi_q:
                return False
            return oracle_gate(lo_q, hi_q)

        def dfs(i: int, S: int) -> None:
            nonlocal best, best_vec
            if best == T:
                return
            self._tick()
            if not suffix_gate(i, S):
                return
            if i == n:
                if side == -1 and S <= T and (best is None or S > best):
                    best, best_vec = S, tuple(x)
                elif side == 1 and S >= T and (best is None or S < best):
                    best, best_vec = S, tuple(x)
                return

            m_i = m[i]
            suffix_max = smax[i + 1]

            # Window of actual counts c for component i worth visiting.
            if side == -1:
                # S + (c - lo)*m_i <= T
                c_hi = min(self.hi[i], self.lo[i] + (T - S) // m_i)
                c_lo = self.lo[i]
                if best is not None:
                    # subtree must be able to beat incumbent even filled full:
                    # S + (c-lo)*m_i + suffix_max > best
                    c_lo = max(c_lo, self.lo[i]
                               + (best - S - suffix_max) // m_i + 1)
            else:
                # S + (c-lo)*m_i + suffix_max >= T
                need = T - S - suffix_max
                c_lo = max(self.lo[i], self.lo[i] + _ceil_div(need, m_i))
                c_hi = self.hi[i]
                if best is not None:
                    # subtree's emptiest mass must stay strictly below best:
                    # S + (c-lo)*m_i < best
                    c_hi = min(c_hi, self.lo[i] + (best - 1 - S) // m_i)

            if c_lo > c_hi:
                return

            # Start at the count whose raw total is closest to T so that the
            # incumbent tightens immediately; then alternate outward.
            ideal = self.lo[i] + (T - S) // m_i
            c0 = c_lo if ideal < c_lo else c_hi if ideal > c_hi else ideal

            def visit(c: int) -> None:
                x[i] = c
                dfs(i + 1, S + (c - self.lo[i]) * m_i)
                x[i] = self.lo[i]

            visit(c0)
            step = 1
            while best != T:
                down = c0 - step
                up = c0 + step
                if down < c_lo and up > c_hi:
                    break
                # For the below side probe smaller counts first (they cannot
                # overshoot); above side symmetrically probes larger counts.
                if side == -1:
                    if down >= c_lo:
                        visit(down)
                    if up <= c_hi:
                        visit(up)
                else:
                    if up <= c_hi:
                        visit(up)
                    if down >= c_lo:
                        visit(down)
                step += 1

        dfs(0, self.base)
        if best is None or best_vec is None:
            return None
        return best, best_vec

    # ------------------------------------------------------------------ #
    # Phase 2: minimum particle count at a fixed exact mass
    # ------------------------------------------------------------------ #

    def min_particles(self, mass: int, max_collect: int) -> dict:
        """Minimum-count vectors attaining the exact ``mass``.

        Returns a dict with the minimum particle count, the *exact* number of
        vectors attaining it (arbitrary precision), up to ``max_collect``
        witness vectors in canonical (id) order, and a truncation flag.
        """
        self.nodes = 0
        R0 = mass - self.base
        if R0 < 0:
            raise ValueError("mass below box minimum")

        # Heavy masses first: minimizing counts then pushes residual onto few
        # large coins and keeps explored counts small.
        order = sorted(range(self.n), key=lambda i: -self.m[i])
        mm = [self.m[i] for i in order]
        cc = [self.cap[i] for i in order]
        n = self.n

        if R0 == 0:
            return {
                "particle_count": sum(self.lo),
                "num_vectors": 1,
                "vectors": [tuple(self.lo)],
                "truncated": False,
            }

        smax = [0] * (n + 1)
        sg = [0] * (n + 2)
        for i in range(n - 1, -1, -1):
            smax[i] = smax[i + 1] + cc[i] * mm[i]
            sg[i] = gcd(mm[i], sg[i + 1])

        if R0 > smax[0] or R0 % sg[0] != 0:
            raise ValueError("mass not attainable")

        # memo[i][R] = (minimum suffix count, number of ways) or _INFEASIBLE.
        memo: list[dict[int, object]] = [dict() for _ in range(n + 1)]

        def solve(i: int, R: int) -> Optional[tuple[int, int]]:
            if R == 0:
                return (0, 1)
            if i == n or R > smax[i] or R % sg[i] != 0:
                return None
            cached = memo[i].get(R, _INFEASIBLE)
            if cached is not _INFEASIBLE:
                return cached  # type: ignore[return-value]
            self._tick()

            m_i, cap_i = mm[i], cc[i]
            g2 = sg[i + 1]

            # Residual R - y*m_i must lie inside [0, smax[i+1]].
            y_hi = min(cap_i, R // m_i)
            y_lo = max(0, _ceil_div(R - smax[i + 1], m_i))
            if y_lo > y_hi:
                memo[i][R] = None
                return None

            cand = _congruence_candidates(y_lo, y_hi, m_i, R, g2)
            best_k: Optional[int] = None
            best_ways = 0
            if cand is not None:
                y, step = cand
                while y <= y_hi:
                    if best_k is not None and y > best_k:
                        break  # suffix counts are non-negative
                    sub = solve(i + 1, R - y * m_i)
                    if sub is not None:
                        k = y + sub[0]
                        if best_k is None or k < best_k:
                            best_k, best_ways = k, sub[1]
                        elif k == best_k:
                            best_ways += sub[1]
                    y += step

            result = None if best_k is None else (best_k, best_ways)
            memo[i][R] = result
            return result

        root = solve(0, R0)
        if root is None:
            raise ValueError("mass not attainable")
        min_extra, num_ways = root

        collected: list[tuple[int, ...]] = []

        def collect(i: int, R: int, prefix: list[int]) -> None:
            if len(collected) >= max_collect:
                return
            if R == 0:
                eff = [0] * n
                for pos, y in enumerate(prefix):
                    eff[pos] = y
                vec = [0] * n
                for pos, idx in enumerate(order):
                    vec[idx] = self.lo[idx] + eff[pos]
                collected.append(tuple(vec))
                return
            entry = memo[i].get(R, _INFEASIBLE)
            if entry is None or entry is _INFEASIBLE:
                return
            target_k = entry[0]
            m_i, cap_i = mm[i], cc[i]
            y_hi = min(cap_i, R // m_i)
            y_lo = max(0, _ceil_div(R - smax[i + 1], m_i))
            cand = _congruence_candidates(y_lo, y_hi, m_i, R, sg[i + 1])
            if cand is None:
                return
            y, step = cand
            while y <= y_hi and y <= target_k:
                sub = solve(i + 1, R - y * m_i)
                if sub is not None and y + sub[0] == target_k:
                    prefix.append(y)
                    collect(i + 1, R - y * m_i, prefix)
                    prefix.pop()
                y += step

        collect(0, R0, [])

        return {
            "particle_count": sum(self.lo) + min_extra,
            "num_vectors": num_ways,
            "vectors": collected,
            "truncated": num_ways > len(collected),
        }

    # ------------------------------------------------------------------ #
    # Top-level driver
    # ------------------------------------------------------------------ #

    def solve(self, max_collect: int) -> dict:
        below = self._extreme(-1)
        above = self._extreme(1)

        extremes: list[tuple[int, tuple[int, ...]]] = []
        if below is not None:
            extremes.append(below)
        if above is not None:
            extremes.append(above)
        if not extremes:
            raise ValueError("empty counting box")  # prevented by validation

        best_dist = min(abs(mass - self.T) for mass, _ in extremes)
        optimal_masses = sorted({mass for mass, _ in extremes
                                 if abs(mass - self.T) == best_dist})
        within = best_dist <= self.tol

        if not within:
            return {
                "status": "unsatisfiable",
                "target": self.T,
                "tolerance": self.tol,
                "within_tolerance": False,
                "best_distance": best_dist,
                "component_order": self.ids,
                "nearest_below": self._witness(below),
                "nearest_above": self._witness(above),
            }

        # A below and an above witness sharing |error| are BOTH optimal
        # masses; the particle-count objective is taken across all of them.
        blocks = []
        for mass in optimal_masses:
            info = self.min_particles(mass, max_collect)
            blocks.append({
                "mass": mass,
                "particle_count": info["particle_count"],
                "num_vectors": info["num_vectors"],
                "vectors": sorted(info["vectors"]),
                "truncated": info["truncated"],
            })

        best_particles = min(b["particle_count"] for b in blocks)
        winners: list[tuple[int, ...]] = []
        num_winners = 0
        truncated = False
        for b in blocks:
            if b["particle_count"] == best_particles:
                winners.extend(b["vectors"])
                num_winners += b["num_vectors"]
                truncated = truncated or b["truncated"]
        winners.sort()

        return {
            "status": "optimal",
            "target": self.T,
            "tolerance": self.tol,
            "within_tolerance": True,
            "best_distance": best_dist,
            "component_order": self.ids,
            "optimal_masses": optimal_masses,
            "particle_count": best_particles,
            "num_optimal_explanations": num_winners,
            "unique": num_winners == 1,
            "truncated_list": truncated or num_winners > len(winners),
            "vectors": winners,
            "nearest_below": self._witness(below),
            "nearest_above": self._witness(above),
        }

    def _witness(self, extreme: Optional[tuple[int, tuple[int, ...]]]) -> Optional[dict]:
        if extreme is None:
            return None
        mass, vec = extreme
        return {
            "total_mass": mass,
            "error": mass - self.T,
            "absolute_error": abs(mass - self.T),
            "particle_count": sum(vec),
            "vector": list(vec),
            "counts": [
                {"id": self.ids[i], "mass": self.m[i], "count": vec[i],
                 "mass_contribution": vec[i] * self.m[i]}
                for i in range(self.n)
            ],
        }


def _congruence_candidates(y_lo: int, y_hi: int, m_i: int, R: int,
                           g2: int) -> Optional[tuple[int, int]]:
    """Smallest y >= y_lo with m_i*y ≡ R (mod g2), together with the step.

    Returns ``(first_y, step)`` or ``None`` when the congruence has no
    solution in the window.  ``g2 <= 1`` imposes no restriction.
    """
    if g2 <= 1:
        return y_lo, 1
    a = m_i % g2
    h = gcd(a, g2)
    if R % h != 0:
        return None
    mod = g2 // h
    if mod == 1:
        return y_lo, 1
    a2 = a // h
    b2 = (R // h) % mod
    inv = pow(a2, -1, mod)
    r0 = (inv * b2) % mod
    first = y_lo + ((r0 - y_lo) % mod)
    if first > y_hi:
        return None
    return first, mod
