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
# Phase-1 grouped searches memoize exact states (i, mass, group-usages); the
# memo is bounded so pathological residual fan-out cannot exhaust memory.
_GROUP_MEMO_CAP = 500_000
# Phase-2 exact-mass DPs likewise bound their memo tables; once the cap is
# reached states are simply recomputed instead of cached (results are
# unchanged; the node budget still bounds total work).
_PHASE2_MEMO_CAP = 500_000


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
                 groups: Optional[list[dict]] = None,
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

        # Optional disjoint component groups with closed count quotas.
        # ``group_of[pos]`` is the group index owning the canonical component
        # at ``pos`` (groups are validated pairwise disjoint) or ``None``.
        id_pos = {cid: i for i, cid in enumerate(self.ids)}
        self.gnames: list[str] = []
        self.gmembers: list[list[int]] = []
        self.glo: list[int] = []
        self.ghi: list[int] = []
        self.group_of: list[Optional[int]] = [None] * self.n
        self.group_base: list[int] = []
        self.group_top: list[int] = []
        if groups:
            for g in groups:
                gi = len(self.gnames)
                self.gnames.append(g["name"])
                self.glo.append(g["min"])
                self.ghi.append(g["max"])
                positions = sorted(id_pos[cid] for cid in g["components"])
                self.gmembers.append(positions)
                for p in positions:
                    self.group_of[p] = gi
                self.group_base.append(sum(self.lo[p] for p in positions))
                self.group_top.append(sum(self.hi[p] for p in positions))
        self.G = len(self.gnames)
        # Effective quota window for the *added* counts y = x - lo.
        self.glo_eff = [self.glo[g] - self.group_base[g]
                        for g in range(self.G)]
        self.ghi_eff = [self.ghi[g] - self.group_base[g]
                        for g in range(self.G)]

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
    # Phase 1 with group quotas
    #
    # The feasibility polyhedron now carries G extra sum constraints in
    # addition to the per-component box, so suffix mass windows alone are not
    # sound: feasibility of a prefix depends on how much of each group quota
    # the prefix has consumed.  The search below therefore threads an exact
    # tuple of per-group added counts, prunes with suffix mass windows plus a
    # group-capacity window, and -- importantly -- optimizes directly inside
    # the constrained set.  Nothing is filtered after an unconstrained solve.
    # ------------------------------------------------------------------ #

    def _greedy_grouped(self, side: int) -> Optional[tuple[int, tuple[int, ...]]]:
        """Group-respecting incumbent seed.

        below: the smallest quota-feasible vector (every group at its lower
        quota, its counts placed on the lightest members; ungrouped components
        at their mins), then greedily filled toward the target without
        exceeding it.  Returns ``None`` when even the smallest feasible vector
        lies above the target.
        above: the largest quota-feasible vector (groups at their upper
        quotas on the heaviest members; ungrouped components at max), then
        greedily trimmed toward the target without dropping below it.
        """
        x = self.lo[:]
        u = [0] * self.G

        if side == -1:
            # Minimal feasible vector: lower-quota counts on lightest members.
            for g in range(self.G):
                need = self.glo_eff[g]
                for p in sorted(self.gmembers[g], key=lambda i: self.m[i]):
                    take = min(self.cap[p], need)
                    x[p] += take
                    u[g] += take
                    need -= take
            total = sum(x[i] * self.m[i] for i in range(self.n))
            if total > self.T:
                return None  # no feasible vector can be at or below target
            # Fill toward T: repeatedly add the heaviest coin that fits.
            for p in sorted(range(self.n), key=lambda i: -self.m[i]):
                g = self.group_of[p]
                room = self.cap[p] - (x[p] - self.lo[p])
                if g is not None:
                    room = min(room, self.ghi_eff[g] - u[g])
                if room <= 0:
                    continue
                take = min(room, (self.T - total) // self.m[p])
                if take > 0:
                    x[p] += take
                    total += take * self.m[p]
                    if g is not None:
                        u[g] += take
            return total, tuple(x)

        # Maximal feasible vector: upper-quota counts on heaviest members.
        for g in range(self.G):
            budget = self.ghi_eff[g]
            for p in sorted(self.gmembers[g], key=lambda i: -self.m[i]):
                take = min(self.cap[p], budget)
                x[p] += take
                u[g] += take
                budget -= take
        # Ungrouped components go to their individual maximums.
        for p in range(self.n):
            if self.group_of[p] is None:
                x[p] = self.hi[p]
        total = sum(x[i] * self.m[i] for i in range(self.n))
        if total < self.T:
            return None  # no feasible vector can be at or above target
        # Trim toward T: remove the heaviest removable coin first.
        for p in sorted(range(self.n), key=lambda i: -self.m[i]):
            g = self.group_of[p]
            removable = x[p] - self.lo[p]
            if g is not None:
                removable = min(removable, u[g] - self.glo_eff[g])
            if removable <= 0:
                continue
            take = min(removable, (total - self.T) // self.m[p])
            if take > 0:
                x[p] -= take
                total -= take * self.m[p]
                if g is not None:
                    u[g] -= take
        if total < self.T:  # greedy overshot; no valid seed from this side
            return None
        return total, tuple(x)

    def _extreme_grouped(self, side: int) -> Optional[tuple[int, tuple[int, ...]]]:
        """Group-quota version of :meth:`_extreme`.

        side = -1 maximizes M <= T; side = +1 minimizes M >= T, over exactly
        the vectors satisfying every group quota simultaneously.
        """
        n, m, T, G = self.n, self.m, self.T, self.G

        # Quick global feasibility of the group windows themselves (validation
        # already guarantees member-bound attainability; this also covers the
        # interaction with the mass sides).
        for g in range(G):
            if self.glo_eff[g] > self.ghi_eff[g]:
                return None

        # Suffix mass window over every component (DFS walks canonical order),
        smax = [0] * (n + 1)
        sg = [0] * (n + 2)
        for i in range(n - 1, -1, -1):
            smax[i] = smax[i + 1] + self.cap[i] * m[i]
            sg[i] = gcd(m[i], sg[i + 1])

        # Per-group suffix tables in canonical DFS order: remaining member
        # capacity, all-full group mass, and lightest-first (mass, cap) runs
        # for exact forced-mass / capped-mass bounds.
        gcap = [[0] * G for _ in range(n + 1)]
        gruns: list[list[list[tuple[int, int]]]] = [
            [[] for _ in range(G)] for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for g in range(G):
                gcap[i][g] = gcap[i + 1][g]
                gruns[i][g] = list(gruns[i + 1][g])
            g = self.group_of[i]
            if g is not None:
                gcap[i][g] += self.cap[i]
                gruns[i][g] = sorted(gruns[i][g] + [(m[i], self.cap[i])])

        seed = self._greedy_grouped(side)
        best: Optional[int] = seed[0] if seed is not None else None
        best_vec: Optional[tuple[int, ...]] = seed[1] if seed is not None else None

        x = self.lo[:]
        u = [0] * G
        memo: set[tuple] = set()

        def suffix_window(i: int) -> Optional[tuple[int, int]]:
            """Group-forced suffix-contribution mass window, or ``None``.

            Returns ``(forced, max_mass)`` where ``forced`` is the mass forced
            by every outstanding group lower quota (lightest coins) and
            ``max_mass`` is the mass still possible under every upper quota
            (all-full suffix after the lightest excess coins are dropped).
            Both bounds concern the *suffix contribution only*.
            """
            forced = 0
            max_mass = smax[i]
            for g in range(G):
                low = self.glo_eff[g] - u[g]
                high = self.ghi_eff[g] - u[g]
                if high < 0 or low > gcap[i][g]:
                    return None
                if low > 0:
                    forced += _run_lightest_sum(gruns[i][g], low)
                excess = gcap[i][g] - high
                if excess > 0:
                    max_mass -= _run_lightest_sum(gruns[i][g], excess)
            return forced, max_mass

        def suffix_feasible(i: int, S: int) -> bool:
            win = suffix_window(i)
            if win is None:
                return False
            lo_q, hi_q = win
            # Merge the target-side/incumbent interval (all expressed as
            # suffix contributions Q with total S + Q).
            if side == -1:
                hi_q = min(hi_q, T - S)
                if best is not None:
                    lo_q = max(lo_q, best + 1 - S)
            else:
                lo_q = max(lo_q, T - S)
                if best is not None:
                    hi_q = min(hi_q, best - 1 - S)
            if lo_q > hi_q or hi_q < 0:
                return False
            if lo_q < 0:
                lo_q = 0
            g = sg[i]
            if g:
                first = lo_q + ((-lo_q) % g)
                if first > hi_q:
                    return False
            return True

        def dfs(i: int, S: int) -> None:
            nonlocal best, best_vec
            if best == T:
                return
            self._tick()
            if not suffix_feasible(i, S):
                return

            if i == n:
                for g in range(G):
                    if not (self.glo_eff[g] <= u[g] <= self.ghi_eff[g]):
                        return
                if side == -1 and S <= T and (best is None or S > best):
                    best, best_vec = S, tuple(x)
                elif side == 1 and S >= T and (best is None or S < best):
                    best, best_vec = S, tuple(x)
                return

            # Exact-state memoization: identical (i, mass, usage) subtrees are
            # equivalent.  The set is bounded so pathological residual fan-out
            # cannot exhaust memory; once full the search simply stops pruning
            # by this memo (node budget still applies).
            key = (i, S, tuple(u))
            if len(memo) < _GROUP_MEMO_CAP:
                if key in memo:
                    return
                memo.add(key)

            m_i = m[i]
            g = self.group_of[i]
            y_cur = x[i] - self.lo[i]
            y_max = self.cap[i]
            # Counts on this component must leave the rest of its group's
            # window reachable with the remaining member positions.
            y_min_group = 0
            if g is not None:
                outstanding_lo = self.glo_eff[g] - u[g]
                y_min_group = max(0, outstanding_lo - gcap[i + 1][g])
                y_max = min(y_max, self.ghi_eff[g] - u[g])
            if y_max < 0 or y_min_group > y_max:
                return

            suffix_max = smax[i + 1]

            if side == -1:
                y_hi = min(y_max, (T - S) // m_i)
                y_lo = y_min_group
                if best is not None:
                    # S + y*m_i + suffix_max must be able to beat best.
                    y_lo = max(y_lo, (best - S - suffix_max) // m_i + 1)
            else:
                need = T - S - suffix_max
                y_lo = max(y_min_group, _ceil_div(need, m_i))
                y_hi = y_max
                if best is not None:
                    y_hi = min(y_hi, (best - 1 - S) // m_i)

            if y_lo > y_hi:
                return

            # Visit the count closest to the target first, alternating outward;
            # the constrained side (below -> smaller y, above -> larger y) goes
            # first so the incumbent tightens quickly.
            ideal = (T - S) // m_i
            y0 = y_lo if ideal < y_lo else y_hi if ideal > y_hi else ideal

            def visit(yv: int) -> None:
                x[i] = self.lo[i] + yv
                if g is not None:
                    u[g] += yv - y_cur
                dfs(i + 1, S + (yv - y_cur) * m_i)
                if g is not None:
                    u[g] -= yv - y_cur
                x[i] = self.lo[i] + y_cur

            visit(y0)
            step = 1
            while best != T:
                down = y0 - step
                up = y0 + step
                if down < y_lo and up > y_hi:
                    break
                if side == -1:
                    if down >= y_lo:
                        visit(down)
                    if up <= y_hi:
                        visit(up)
                else:
                    if up <= y_hi:
                        visit(up)
                    if down >= y_lo:
                        visit(down)
                step += 1

        dfs(0, self.base)
        if best is None or best_vec is None:
            return None
        return best, best_vec

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
        memo_size = 0

        def cache_put(i: int, R: int, value: object) -> None:
            nonlocal memo_size
            if R not in memo[i]:
                if memo_size >= _PHASE2_MEMO_CAP:
                    return  # stop caching; uncached states are recomputed
                memo_size += 1
            memo[i][R] = value

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
                cache_put(i, R, None)
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
            cache_put(i, R, result)
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
            if entry is None:
                return
            if entry is _INFEASIBLE:
                # Memo cap dropped this state; recompute it on demand.
                entry = solve(i, R)
                if entry is None:
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
    # Phase 2 with group quotas: exact minimum particle count at one mass
    #
    # The memoized suffix state is extended with the vector of per-group
    # added counts already consumed, (i, R, u); transitions for a component
    # that belongs to a group additionally clip the coin window to the
    # outstanding quota interval.  The optimum is therefore computed *within*
    # the constrained set directly -- it is never an unconstrained answer
    # filtered afterwards.
    # ------------------------------------------------------------------ #

    def min_particles_grouped(self, mass: int, max_collect: int) -> dict:
        """Group-quota version of :meth:`min_particles`."""
        self.nodes = 0
        R0 = mass - self.base
        if R0 < 0:
            raise ValueError("mass below box minimum")

        n, G = self.n, self.G
        order = sorted(range(n), key=lambda i: -self.m[i])
        mm = [self.m[i] for i in order]
        cc = [self.cap[i] for i in order]
        og = [self.group_of[i] for i in order]

        # Per ordered suffix position i and group g: total remaining member
        # capacity, the member (mass, cap) pairs sorted lightest-first, and
        # cumulative count/mass runs for exact lightest-k mass lookups.
        suf_cap = [[0] * G for _ in range(n + 1)]
        suf_runs: list[list[list[tuple[int, int]]]] = [
            [[] for _ in range(G)] for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for g in range(G):
                suf_cap[i][g] = suf_cap[i + 1][g]
                suf_runs[i][g] = list(suf_runs[i + 1][g])
            g = og[i]
            if g is not None:
                suf_cap[i][g] += cc[i]
                suf_runs[i][g] = sorted(suf_runs[i][g] + [(mm[i], cc[i])])

        smax = [0] * (n + 1)
        sg = [0] * (n + 2)
        for i in range(n - 1, -1, -1):
            smax[i] = smax[i + 1] + cc[i] * mm[i]
            sg[i] = gcd(mm[i], sg[i + 1])

        def lightest_mass(i: int, g: int, k: int) -> int:
            """Sum of the k lightest coins available to group g in suffix i."""
            if k <= 0:
                return 0
            return _run_lightest_sum(suf_runs[i][g], k)

        def _bounds_feasible(i: int, R: int, u: tuple[int, ...]) -> bool:
            """Group count windows and forced suffix-mass window at state."""
            if R > smax[i] or R % sg[i] != 0:
                return False
            min_forced = 0
            max_allowed = smax[i]  # shrunk per group below
            for g in range(G):
                low = self.glo_eff[g] - u[g]
                high = self.ghi_eff[g] - u[g]
                if high < 0 or low > suf_cap[i][g]:
                    return False
                if low > 0:
                    # The lower quota forces at least this much suffix mass.
                    min_forced += lightest_mass(i, g, low)
                excess = suf_cap[i][g] - high
                if excess > 0:
                    # Capping the group at 'high' coins: the lightest excess
                    # coins must be dropped from the all-full suffix mass.
                    max_allowed -= lightest_mass(i, g, excess)
            return min_forced <= R <= max_allowed

        zero = (0,) * G
        if R0 == 0:
            ok = all(self.glo_eff[g] <= 0 <= self.ghi_eff[g]
                     for g in range(G))
            if not ok:
                raise ValueError("mass not attainable under group quotas")
            return {
                "particle_count": sum(self.lo),
                "num_vectors": 1,
                "vectors": [tuple(self.lo)],
                "truncated": False,
            }

        if not _bounds_feasible(0, R0, zero):
            raise ValueError("mass not attainable under group quotas")

        memo: dict[tuple, Optional[tuple[int, int]]] = {}
        INFEAS = _INFEASIBLE
        memo_full = False

        def cache_put(key: tuple, value: Optional[tuple[int, int]]) -> None:
            nonlocal memo_full
            if memo_full:
                return
            if len(memo) >= _PHASE2_MEMO_CAP:
                memo_full = True
                return
            memo[key] = value

        def solve(i: int, R: int,
                  u: tuple[int, ...]) -> Optional[tuple[int, int]]:
            if R == 0:
                # No further coins: every lower quota must already be met.
                for g in range(G):
                    if not (self.glo_eff[g] <= u[g] <= self.ghi_eff[g]):
                        return None
                return (0, 1)
            if i == n or not _bounds_feasible(i, R, u):
                return None
            key = (i, R, u)
            cached = memo.get(key, INFEAS)
            if cached is not INFEAS:
                return cached
            self._tick()

            m_i, cap_i = mm[i], cc[i]
            g = og[i]
            y_hi = min(cap_i, R // m_i)
            y_lo = max(0, _ceil_div(R - smax[i + 1], m_i))
            if g is not None:
                # Upper quota and the need to leave enough suffix capacity
                # for the outstanding lower quota bound this coin's window.
                y_hi = min(y_hi, self.ghi_eff[g] - u[g])
                y_lo = max(y_lo,
                           self.glo_eff[g] - u[g] - suf_cap[i + 1][g])
            if y_lo > y_hi:
                cache_put(key, None)
                return None

            cand = _congruence_candidates(y_lo, y_hi, m_i, R, sg[i + 1])
            best_k: Optional[int] = None
            best_ways = 0
            if cand is not None:
                y, step = cand
                while y <= y_hi:
                    if best_k is not None and y > best_k:
                        break
                    if g is not None:
                        u2 = u[:g] + (u[g] + y,) + u[g + 1:]
                    else:
                        u2 = u
                    sub = solve(i + 1, R - y * m_i, u2)
                    if sub is not None:
                        k = y + sub[0]
                        if best_k is None or k < best_k:
                            best_k, best_ways = k, sub[1]
                        elif k == best_k:
                            best_ways += sub[1]
                    y += step

            result = None if best_k is None else (best_k, best_ways)
            cache_put(key, result)
            return result

        root = solve(0, R0, zero)
        if root is None:
            raise ValueError("mass not attainable under group quotas")
        min_extra, num_ways = root

        collected: list[tuple[int, ...]] = []

        def collect(i: int, R: int, u: tuple[int, ...],
                    prefix: list[int]) -> None:
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
            entry = memo.get((i, R, u), INFEAS)
            if entry is None:
                return  # cached infeasible state
            if entry is INFEAS:
                # Memo cap dropped this state; recompute it on demand.
                entry = solve(i, R, u)
                if entry is None:
                    return
            target_k = entry[0]
            m_i, cap_i = mm[i], cc[i]
            g = og[i]
            y_hi = min(cap_i, R // m_i)
            y_lo = max(0, _ceil_div(R - smax[i + 1], m_i))
            if g is not None:
                y_hi = min(y_hi, self.ghi_eff[g] - u[g])
                y_lo = max(y_lo,
                           self.glo_eff[g] - u[g] - suf_cap[i + 1][g])
            cand = _congruence_candidates(y_lo, y_hi, m_i, R, sg[i + 1])
            if cand is None:
                return
            y, step = cand
            while y <= y_hi and y <= target_k:
                if g is not None:
                    u2 = u[:g] + (u[g] + y,) + u[g + 1:]
                else:
                    u2 = u
                sub = solve(i + 1, R - y * m_i, u2)
                if sub is not None and y + sub[0] == target_k:
                    prefix.append(y)
                    collect(i + 1, R - y * m_i, u2, prefix)
                    prefix.pop()
                y += step

        collect(0, R0, zero, [])

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
        if self.G:
            return self._solve_grouped(max_collect)
        return self._solve_plain(max_collect)

    def _solve_plain(self, max_collect: int) -> dict:
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

    def _group_totals(self, vec: tuple[int, ...] | list[int]) -> list[int]:
        return [sum(vec[p] for p in members) for members in self.gmembers]

    def _solve_grouped(self, max_collect: int) -> dict:
        below = self._extreme_grouped(-1)
        above = self._extreme_grouped(1)

        extremes: list[tuple[int, tuple[int, ...]]] = []
        if below is not None:
            extremes.append(below)
        if above is not None:
            extremes.append(above)
        if not extremes:
            # Validation guarantees a non-empty feasible set; reaching here
            # means the quota windows are jointly inconsistent.
            raise ValueError("no vector satisfies the group quotas")

        best_dist = min(abs(mass - self.T) for mass, _ in extremes)
        optimal_masses = sorted({mass for mass, _ in extremes
                                 if abs(mass - self.T) == best_dist})

        if best_dist > self.tol:
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

        blocks = []
        for mass in optimal_masses:
            info = self.min_particles_grouped(mass, max_collect)
            blocks.append({
                "mass": mass,
                "particle_count": info["particle_count"],
                "num_vectors": info["num_vectors"],
                "vectors": sorted(info["vectors"]),
                "truncated": info["truncated"],
            })
        return self._grouped_result(
            blocks, best_dist, optimal_masses, below, above)

    def _grouped_result(self, blocks: list[dict], best_dist: int,
                        optimal_masses: list[int],
                        below: Optional[tuple[int, tuple[int, ...]]],
                        above: Optional[tuple[int, tuple[int, ...]]]) -> dict:
        """Assemble the grouped solver result from per-mass optimal blocks."""
        if best_dist > self.tol:
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
        doc = {
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
        if self.G:
            totals = self._group_totals(vec)
            doc["group_totals"] = [
                {"name": self.gnames[g],
                 "components": [self.ids[p] for p in self.gmembers[g]],
                 "total": totals[g],
                 "min": self.glo[g], "max": self.ghi[g],
                 "within_quota": self.glo[g] <= totals[g] <= self.ghi[g]}
                for g in range(self.G)
            ]
        return doc


def _run_lightest_sum(runs: list[tuple[int, int]], k: int) -> int:
    """Sum of the ``k`` lightest coins from ``(mass, cap)`` runs.

    ``runs`` is sorted by non-decreasing mass.  Group membership has at most
    16 components, so a linear scan is exact and cheap; no count interval is
    materialized.
    """
    total = 0
    remaining = k
    for mass, cap in runs:
        take = cap if cap < remaining else remaining
        total += take * mass
        remaining -= take
        if remaining == 0:
            break
    return total


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
