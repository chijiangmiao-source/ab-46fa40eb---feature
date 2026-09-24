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

Group quotas
------------
An optional review constraint declares pairwise disjoint groups of component
ids together with a closed interval ``[qlo, qhi]`` on the *total* count of the
group's members::

    qlo_g <= sum_{i in group g} x_i <= qhi_g

The quotas are enforced *inside* both search phases: suffix mass windows are
tightened by the cheapest/dearest extra counts each quota still allows, and
branching windows respect each group's running total.  Feasible vectors are
therefore never obtained by solving the unconstrained problem and filtering
afterwards; when no vector exists, the nearest below/above witnesses are
themselves quota-feasible.
"""

from __future__ import annotations

from bisect import bisect_left
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


@dataclass(frozen=True)
class GroupSpec:
    """A quota over the total count of a disjoint set of component ids."""
    members: tuple[str, ...]
    qlo: int
    qhi: int
    name: Optional[str] = None


_INFEASIBLE = object()


def _ceil_div(a: int, b: int) -> int:
    """ceil(a / b) for b > 0, exact for possibly-negative a."""
    return -((-a) // b)


class _CumulativeFill:
    """Precomputed fill curve over a fixed set of (mass, cap) coins.

    ``mass_for(e)`` is the exact mass of ``e`` indistinguishable extra counts
    when they are placed on the pre-sorted coins (cheapest-first for a minimum
    mass, dearest-first for a maximum mass).  Built without expanding any
    counting interval: a prefix-sum lookup with O(log coins) cost.
    """

    __slots__ = ("masses", "caps", "cum_cap", "cum_mass")

    def __init__(self, pairs: list[tuple[int, int]]):
        # pairs must already be sorted in the desired filling direction.
        self.masses = [p[0] for p in pairs]
        self.caps = [p[1] for p in pairs]
        self.cum_cap = [0]
        self.cum_mass = [0]
        for mass, cap in pairs:
            self.cum_cap.append(self.cum_cap[-1] + cap)
            self.cum_mass.append(self.cum_mass[-1] + mass * cap)

    def mass_for(self, e: int) -> Optional[int]:
        if e <= 0:
            return 0
        if e > self.cum_cap[-1]:
            return None
        j = bisect_left(self.cum_cap, e)  # first prefix reaching e
        prev = j - 1
        return self.cum_mass[prev] + (e - self.cum_cap[prev]) * self.masses[prev]


class _GroupSnapshot:
    """Quota-relevant view of one group's still-unvisited coordinates."""

    __slots__ = ("positions", "lo_sum", "hi_sum", "cap_sum", "cheap", "dear")

    def __init__(self, positions: list[int], lo: list[int], hi: list[int],
                 cap: list[int], m: list[int]):
        self.positions = positions
        self.lo_sum = sum(lo[p] for p in positions)
        self.hi_sum = sum(hi[p] for p in positions)
        self.cap_sum = sum(cap[p] for p in positions)
        pairs = [(m[p], cap[p]) for p in positions]
        self.cheap = _CumulativeFill(sorted(pairs, key=lambda pm: pm[0]))
        self.dear = _CumulativeFill(sorted(pairs, key=lambda pm: -pm[0]))


class Solver:
    def __init__(self, components: list[Component], target: int, tolerance: int,
                 groups: Optional[list[GroupSpec]] = None,
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

        # Disjoint group membership in canonical coordinates.
        self.group_of = [-1] * self.n
        self.gmembers: list[list[int]] = []
        self.glo: list[int] = []
        self.ghi: list[int] = []
        self.gnames: list[Optional[str]] = []
        for spec in groups or []:
            idxs = sorted(self.ids.index(mid) for mid in spec.members)
            g = len(self.gmembers)
            for p in idxs:
                self.group_of[p] = g
            self.gmembers.append(idxs)
            self.glo.append(spec.qlo)
            self.ghi.append(spec.qhi)
            self.gnames.append(spec.name)
        self.G = len(self.gmembers)

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes > self.node_budget:
            raise BudgetExceeded(f"search visited {self.nodes} nodes")

    def _build_views(self, order: list[int]):
        """Per-depth group snapshots for coordinates not yet visited.

        ``snaps[d][g]`` describes group g's members that occur at or after
        recursion depth d in ``order``; ``ng_max[d]`` is the largest extra
        mass attainable by non-group coordinates at or after d.
        """
        remaining = [list(members) for members in self.gmembers]
        snaps: list[list[_GroupSnapshot]] = []
        for d in range(self.n + 1):
            view = [_GroupSnapshot(remaining[g], self.lo, self.hi,
                                   self.cap, self.m)
                    for g in range(self.G)]
            snaps.append(view)
            if d < self.n:
                p = order[d]
                g = self.group_of[p]
                if g >= 0:
                    remaining[g] = [q for q in remaining[g] if q != p]

        ng_max = [0] * (self.n + 2)
        for d in range(self.n - 1, -1, -1):
            p = order[d]
            extra = self.cap[p] * self.m[p] if self.group_of[p] < 0 else 0
            ng_max[d] = ng_max[d + 1] + extra
        return snaps, ng_max

    def _group_suffix_window(self, view: list[_GroupSnapshot],
                             gtot: list[int]) -> Optional[tuple[int, int]]:
        """Sound [min, max] extra-mass window forced by quotas on the suffix.

        ``gtot[g]`` is the actual count already fixed on visited members of
        group g.  Returns ``None`` when no completion of the quotas exists.
        """
        lo_mass = hi_mass = 0
        for g in range(self.G):
            snap = view[g]
            fixed = gtot[g]
            if fixed + snap.hi_sum < self.glo[g]:
                return None  # even filling the rest cannot reach qlo
            if fixed + snap.lo_sum > self.ghi[g]:
                return None  # even the emptiest rest exceeds qhi
            e_min = max(0, self.glo[g] - fixed - snap.lo_sum)
            e_max = min(snap.cap_sum, self.ghi[g] - fixed - snap.lo_sum)
            cheap = snap.cheap.mass_for(e_min)
            dear = snap.dear.mass_for(e_max)
            if cheap is None or dear is None or cheap > dear:
                return None
            lo_mass += cheap
            hi_mass += dear
        return lo_mass, hi_mass

    # ------------------------------------------------------------------ #
    # Quota-feasible seed vectors
    # ------------------------------------------------------------------ #

    def _feasible_seeds(self) -> Optional[tuple[list[int], list[int]]]:
        """Quota-feasible minimum-mass and maximum-mass vectors.

        Returns ``None`` when a quota interval is statically disjoint from the
        members' own count bounds (defensive; request validation rejects such
        declarations before the solver is constructed).
        """
        x_lo = self.lo[:]
        x_hi = self.hi[:]
        for g, members in enumerate(self.gmembers):
            lo_sum = sum(self.lo[p] for p in members)
            hi_sum = sum(self.hi[p] for p in members)
            if self.glo[g] > hi_sum or self.ghi[g] < lo_sum:
                return None
            # Lift the all-lo vector to qlo placing extras on cheapest coins.
            need = self.glo[g] - lo_sum
            for p in sorted(members, key=lambda q: self.m[q]):
                if need <= 0:
                    break
                add = min(need, self.cap[p])
                x_lo[p] += add
                need -= add
            if need > 0:
                return None
            # Trim the all-hi vector to qhi removing cheapest coins first,
            # which retains the largest possible total mass.
            excess = hi_sum - self.ghi[g]
            for p in sorted(members, key=lambda q: self.m[q]):
                if excess <= 0:
                    break
                dec = min(excess, x_hi[p] - self.lo[p])
                x_hi[p] -= dec
                excess -= dec
            if excess > 0:
                return None
        return x_lo, x_hi

    def _group_totals(self, x: list[int]) -> list[int]:
        totals = [0] * self.G
        for g, members in enumerate(self.gmembers):
            totals[g] = sum(x[p] for p in members)
        return totals

    # ------------------------------------------------------------------ #
    # Phase 1: nearest reachable mass on each side of the target
    # ------------------------------------------------------------------ #

    def _greedy_below(self) -> Optional[tuple[int, tuple[int, ...]]]:
        """Largest-ish quota-feasible mass <= T (incumbent seed only).

        Two quota-feasible anchors are combined: the global feasible *maximum*
        vector (trimmed down toward T by removing cheapest coins first) and
        the global feasible *minimum* vector (filled up toward T by adding
        dearest coins first).  When the target lies at or above the feasible
        maximum the first anchor is already the exact answer, so no search is
        needed to prove it.
        """
        seeds = self._feasible_seeds()
        if seeds is None:
            return None
        x_lo, x_hi = seeds
        T = self.T
        candidates: list[tuple[int, tuple[int, ...]]] = []

        total_hi = sum(x_hi[i] * self.m[i] for i in range(self.n))
        if total_hi <= T:
            return total_hi, tuple(x_hi)  # global feasible maximum
        x = x_hi[:]
        gtot = self._group_totals(x)
        total = total_hi
        for j in sorted(range(self.n), key=lambda i: self.m[i]):
            room = x[j] - self.lo[j]
            gj = self.group_of[j]
            if gj >= 0:
                room = min(room, gtot[gj] - self.glo[gj])
            dec = min(room, (total - T) // self.m[j])
            if dec:
                x[j] -= dec
                total -= dec * self.m[j]
                if gj >= 0:
                    gtot[gj] -= dec
        if total <= T:
            candidates.append((total, tuple(x)))

        x = x_lo[:]
        total = sum(x[i] * self.m[i] for i in range(self.n))
        if total <= T:
            gtot = self._group_totals(x)
            for j in sorted(range(self.n), key=lambda i: -self.m[i]):
                gj = self.group_of[j]
                room = self.hi[j] - x[j]
                if gj >= 0:
                    room = min(room, self.ghi[gj] - gtot[gj])
                inc = min(room, (T - total) // self.m[j])
                if inc:
                    x[j] += inc
                    total += inc * self.m[j]
                    if gj >= 0:
                        gtot[gj] += inc
            candidates.append((total, tuple(x)))

        return max(candidates, default=None)

    def _greedy_above(self) -> Optional[tuple[int, tuple[int, ...]]]:
        """Smallest-ish quota-feasible mass >= T (incumbent seed only).

        Mirror of :meth:`_greedy_below`: the global feasible minimum is filled
        up toward T by adding cheapest coins first, and the global feasible
        maximum is trimmed down by removing dearest coins first.  A target at
        or below the feasible minimum answers immediately.
        """
        seeds = self._feasible_seeds()
        if seeds is None:
            return None
        x_lo, x_hi = seeds
        T = self.T
        candidates: list[tuple[int, tuple[int, ...]]] = []

        total_lo = sum(x_lo[i] * self.m[i] for i in range(self.n))
        if total_lo >= T:
            return total_lo, tuple(x_lo)  # global feasible minimum
        x = x_lo[:]
        gtot = self._group_totals(x)
        total = total_lo
        for j in sorted(range(self.n), key=lambda i: self.m[i]):
            gj = self.group_of[j]
            room = self.hi[j] - x[j]
            if gj >= 0:
                room = min(room, self.ghi[gj] - gtot[gj])
            inc = min(room, _ceil_div(T - total, self.m[j]))
            if inc:
                x[j] += inc
                total += inc * self.m[j]
                if gj >= 0:
                    gtot[gj] += inc
        if total >= T:
            candidates.append((total, tuple(x)))

        x = x_hi[:]
        total = sum(x[i] * self.m[i] for i in range(self.n))
        if total >= T:
            gtot = self._group_totals(x)
            for j in sorted(range(self.n), key=lambda i: -self.m[i]):
                gj = self.group_of[j]
                room = x[j] - self.lo[j]
                if gj >= 0:
                    room = min(room, gtot[gj] - self.glo[gj])
                dec = min(room, (total - T) // self.m[j])
                if dec:
                    x[j] -= dec
                    total -= dec * self.m[j]
                    if gj >= 0:
                        gtot[gj] -= dec
            candidates.append((total, tuple(x)))

        return min(candidates, default=None)

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
        exists on that side of the target.  Only quota-feasible vectors are
        visited when group quotas are declared.
        """
        n, m, T = self.n, self.m, self.T

        # Suffix information over *effective* counts y_i = x_i - lo_i.
        smax = [0] * (n + 1)
        sg = [0] * (n + 2)
        for i in range(n - 1, -1, -1):
            smax[i] = smax[i + 1] + self.cap[i] * m[i]
            sg[i] = gcd(m[i], sg[i + 1])

        order = list(range(n))  # canonical id order in phase 1
        views, ng_max = self._build_views(order)

        seed = self._greedy_below() if side == -1 else self._greedy_above()
        best: Optional[int] = seed[0] if seed is not None else None
        best_vec: Optional[tuple[int, ...]] = seed[1] if seed is not None else None
        if best == T:
            return best, best_vec  # type: ignore[return-value]

        om: int = oracle[1] if oracle else 0
        od: Optional[list[int]] = oracle[2] if oracle else None
        x = self.lo[:]
        gtot = [0] * self.G

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
            """Prune node (i, S): quota window, suffix range, gcd, incumbent."""
            nonlocal best, best_vec
            window = self._group_suffix_window(views[i], gtot)
            if window is None:
                return False
            g_min, g_max_window = window
            suffix_max = ng_max[i] + g_max_window
            g = sg[i]
            if side == -1:
                hi_q = min(suffix_max, T - S)
                if hi_q < g_min:
                    return False  # quotas force the suffix above T
                q = hi_q if g == 0 else hi_q - (hi_q % g)
                if q < g_min:
                    return False
                if best is not None and S + q <= best:
                    return False
                lo_q = max(g_min, best + 1 - S) if best is not None else g_min
                if lo_q > hi_q:
                    return False
                return oracle_gate(lo_q, hi_q)
            lo_q = max(T - S, g_min)
            if lo_q > suffix_max:
                return False  # even a full suffix cannot reach T
            if g == 0:
                q = lo_q
            else:
                q = lo_q + ((-lo_q) % g)
            if q > suffix_max:
                return False
            if best is not None and S + q >= best:
                return False
            hi_q = min(suffix_max, best - 1 - S) if best is not None else suffix_max
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
            # Loose all-caps suffix maximum (incumbent tightening only; the
            # child's quota gate remains exact, so overestimation is sound).
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

            # Intersect the quota window for this coordinate's group.
            gi = self.group_of[i]
            if gi >= 0:
                after = views[i + 1][gi]  # unvisited group-mates of i
                fixed = gtot[gi]
                c_lo = max(c_lo, self.glo[gi] - fixed - after.hi_sum)
                c_hi = min(c_hi, self.ghi[gi] - fixed - after.lo_sum)

            if c_lo > c_hi:
                return

            # Start at the count whose raw total is closest to T so that the
            # incumbent tightens immediately; then alternate outward.
            ideal = self.lo[i] + (T - S) // m_i
            c0 = c_lo if ideal < c_lo else c_hi if ideal > c_hi else ideal

            def visit(c: int) -> None:
                x[i] = c
                if gi >= 0:
                    gtot[gi] += c
                dfs(i + 1, S + (c - self.lo[i]) * m_i)
                if gi >= 0:
                    gtot[gi] -= c
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
        """Minimum-count quota-feasible vectors attaining exact ``mass``.

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

        smax = [0] * (n + 1)
        sg = [0] * (n + 2)
        for i in range(n - 1, -1, -1):
            smax[i] = smax[i + 1] + cc[i] * mm[i]
            sg[i] = gcd(mm[i], sg[i + 1])

        views, ng_max = self._build_views(order)

        def root_window() -> Optional[tuple[int, int]]:
            return self._group_suffix_window(views[0], [0] * self.G)

        rw = root_window()
        if rw is None or R0 < rw[0]:
            raise ValueError("mass not attainable")
        if R0 > ng_max[0] + rw[1] or (sg[0] and R0 % sg[0] != 0):
            raise ValueError("mass not attainable")

        # memo[i][(R, gtot_tuple)] = (minimum suffix count, ways) or None.
        memo: list[dict[tuple[int, tuple[int, ...]], object]] = [
            dict() for _ in range(n + 1)]

        def solve(i: int, R: int, gtot: list[int]) -> Optional[tuple[int, int]]:
            key = (R, tuple(gtot))
            window = self._group_suffix_window(views[i], gtot)
            if window is None:
                memo[i][key] = None
                return None
            gmin, gmax = window
            if R < gmin or R > ng_max[i] + gmax:
                memo[i][key] = None
                return None
            if R == 0:
                # All remaining effective counts are zero; feasible only when
                # leaving every unvisited group member at lo still meets quota.
                return (0, 1) if gmin == 0 else None
            if i == n or (sg[i] and R % sg[i] != 0):
                return None
            cached = memo[i].get(key, _INFEASIBLE)
            if cached is not _INFEASIBLE:
                return cached  # type: ignore[return-value]
            self._tick()

            idx = order[i]
            m_i, cap_i = mm[i], cc[i]
            g2 = sg[i + 1]

            # Residual R - y*m_i must lie inside [0, smax[i+1]].
            y_hi = min(cap_i, R // m_i)
            y_lo = max(0, _ceil_div(R - smax[i + 1], m_i))

            gi = self.group_of[idx]
            if gi >= 0:
                after = views[i + 1][gi]  # group-mates visited after i
                fixed = gtot[gi]
                actual = self.lo[idx]
                y_lo = max(y_lo, self.glo[gi] - fixed - after.hi_sum - actual)
                y_hi = min(y_hi, self.ghi[gi] - fixed - after.lo_sum - actual)
            if y_lo > y_hi:
                memo[i][key] = None
                return None

            cand = _congruence_candidates(y_lo, y_hi, m_i, R, g2)
            best_k: Optional[int] = None
            best_ways = 0
            if cand is not None:
                y, step = cand
                while y <= y_hi:
                    if best_k is not None and y > best_k:
                        break  # suffix counts are non-negative
                    if gi >= 0:
                        gtot[gi] += self.lo[idx] + y
                    sub = solve(i + 1, R - y * m_i, gtot)
                    if gi >= 0:
                        gtot[gi] -= self.lo[idx] + y
                    if sub is not None:
                        k = y + sub[0]
                        if best_k is None or k < best_k:
                            best_k, best_ways = k, sub[1]
                        elif k == best_k:
                            best_ways += sub[1]
                    y += step

            result = None if best_k is None else (best_k, best_ways)
            memo[i][key] = result
            return result

        gtot0 = [0] * self.G
        root = solve(0, R0, gtot0)
        if root is None:
            raise ValueError("mass not attainable")
        min_extra, num_ways = root

        collected: list[tuple[int, ...]] = []

        def collect(i: int, R: int, gtot: list[int],
                    prefix: list[int]) -> None:
            if len(collected) >= max_collect:
                return
            if R == 0:
                vec = [0] * n
                for pos, idx in enumerate(order):
                    y = prefix[pos] if pos < len(prefix) else 0
                    vec[idx] = self.lo[idx] + y
                collected.append(tuple(vec))
                return
            key = (R, tuple(gtot))
            entry = memo[i].get(key, _INFEASIBLE)
            if entry is None or entry is _INFEASIBLE:
                return
            target_k = entry[0]
            idx = order[i]
            m_i, cap_i = mm[i], cc[i]
            y_hi = min(cap_i, R // m_i)
            y_lo = max(0, _ceil_div(R - smax[i + 1], m_i))
            gi = self.group_of[idx]
            if gi >= 0:
                after = views[i + 1][gi]
                fixed = gtot[gi]
                actual = self.lo[idx]
                y_lo = max(y_lo, self.glo[gi] - fixed - after.hi_sum - actual)
                y_hi = min(y_hi, self.ghi[gi] - fixed - after.lo_sum - actual)
            cand = _congruence_candidates(y_lo, y_hi, m_i, R, sg[i + 1])
            if cand is None:
                return
            y, step = cand
            while y <= y_hi and y <= target_k:
                if gi >= 0:
                    gtot[gi] += self.lo[idx] + y
                sub = solve(i + 1, R - y * m_i, gtot)
                feasible = sub is not None and y + sub[0] == target_k
                if feasible:
                    prefix.append(y)
                    collect(i + 1, R - y * m_i, gtot, prefix)
                    prefix.pop()
                if gi >= 0:
                    gtot[gi] -= self.lo[idx] + y
                y += step

        collect(0, R0, [0] * self.G, [])

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
            raise ValueError("empty or quota-infeasible counting box")

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
