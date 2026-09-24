"""Application service: turn validated requests into response documents."""

from __future__ import annotations

import os
from typing import Any

from .errors import RequestError, ValidationError
from .solver import BudgetExceeded, Component, GroupSpec, Solver
from .validation import MAX_BOUND, parse_request

DEFAULT_MAX_EXPLANATIONS = 100
HARD_MAX_EXPLANATIONS = 10_000


def _node_budget() -> int:
    raw = os.environ.get("SOLVER_NODE_BUDGET", "5000000")
    try:
        val = int(raw)
        return max(1_000, val)
    except ValueError:
        return 5_000_000


def _counts_doc(order: list[str], meta: dict[str, dict[str, int]],
                vec: list[int] | tuple[int, ...]) -> list[dict[str, int]]:
    return [
        {
            "id": cid,
            "mass": meta[cid]["mass"],
            "min": meta[cid]["min"],
            "max": meta[cid]["max"],
            "count": vec[pos],
            "mass_contribution": vec[pos] * meta[cid]["mass"],
        }
        for pos, cid in enumerate(order)
    ]


def _explanation(order: list[str], meta: dict[str, dict[str, int]],
                 mass: int, particle_count: int, target: int,
                 vec: tuple[int, ...]) -> dict[str, Any]:
    counts = _counts_doc(order, meta, vec)
    recomputed = sum(c["mass_contribution"] for c in counts)
    return {
        "total_mass": mass,
        "error": mass - target,                 # signed, in micro-daltons
        "absolute_error": abs(mass - target),
        "particle_count": particle_count,
        "counts": counts,
        "recomputed_total_mass": recomputed,    # equals total_mass exactly
    }


def _witness_doc(order: list[str], meta: dict[str, dict[str, int]],
                 witness: dict[str, Any] | None,
                 target: int,
                 groups: list[dict[str, Any]] | None = None
                 ) -> dict[str, Any] | None:
    if witness is None:
        return None
    doc = {
        "total_mass": witness["total_mass"],
        "error": witness["error"],
        "absolute_error": witness["absolute_error"],
        "particle_count": witness["particle_count"],
        "counts": _counts_doc(order, meta, witness["vector"]),
        "recomputed_total_mass": witness["total_mass"],
        "side": "below" if witness["error"] <= 0 else "above",
    }
    if groups is not None:
        doc["group_counts"] = [
            _group_count_doc(grp, order, witness["vector"]) for grp in groups]
    return doc


def _group_count_doc(grp: dict[str, Any], order: list[str],
                     vec: list[int] | tuple[int, ...]) -> dict[str, Any]:
    """Directly recomputable statement of one group's realized total."""
    total = sum(vec[order.index(mid)] for mid in grp["members"])
    return {
        "name": grp["name"],
        "members": list(grp["members"]),
        "min": grp["min"],
        "max": grp["max"],
        "count": total,
        "satisfied": grp["min"] <= total <= grp["max"],
    }


def invert(body: Any) -> tuple[int, dict[str, Any]]:
    """Execute one inversion. Returns (http_status, response_body)."""
    parsed = parse_request(body)

    max_collect = DEFAULT_MAX_EXPLANATIONS
    if isinstance(body, dict) and "max_explanations" in body:
        raw = body.get("max_explanations")
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise RequestError([ValidationError(
                "invalid_type",
                "'max_explanations' must be a positive integer",
                "/max_explanations")])
        max_collect = min(raw, HARD_MAX_EXPLANATIONS)

    components = [
        Component(id=c["id"], mass=c["mass"], lo=c["min"], hi=c["max"])
        for c in parsed["components"]
    ]
    meta = {
        c.id: {"mass": c.mass, "min": c.lo, "max": c.hi}
        for c in components
    }

    groups = parsed.get("groups") or []
    group_specs = [
        GroupSpec(members=tuple(grp["members"]), qlo=grp["min"],
                  qhi=grp["max"], name=grp["name"])
        for grp in groups
    ]

    solver = Solver(components, parsed["target"], parsed["tolerance"],
                    groups=group_specs, node_budget=_node_budget())
    try:
        result = solver.solve(max_collect)
    except BudgetExceeded as exc:
        return 422, {
            "error": {
                "code": "search_budget_exceeded",
                "message": (f"{exc}; narrow the bounds/target or raise "
                            "SOLVER_NODE_BUDGET"),
            }
        }
    except ValueError:
        # Defensive: request validation already proves every quota interval is
        # reachable, so reaching here means an internally inconsistent box.
        return 422, {
            "error": {
                "code": "quota_infeasible",
                "message": ("no count vector satisfies the declared group "
                            "quotas simultaneously"),
            }
        }

    order = result["component_order"]
    target = result["target"]
    group_arg = groups if groups else None

    common: dict[str, Any] = {
        "target": target,
        "tolerance": result["tolerance"],
        "mass_unit": "micro_dalton",
        "count_bounds_hard_max": MAX_BOUND,
        "component_order": order,
        "nearest_below": _witness_doc(order, meta, result["nearest_below"],
                                      target, group_arg),
        "nearest_above": _witness_doc(order, meta, result["nearest_above"],
                                      target, group_arg),
    }
    if groups:
        common["groups"] = [
            {"name": grp["name"], "members": list(grp["members"]),
             "min": grp["min"], "max": grp["max"]}
            for grp in groups
        ]

    if result["status"] == "unsatisfiable":
        message = ("no reachable total mass lies within the requested "
                   "tolerance; the closest attainable witnesses below "
                   "and above the target are provided")
        if groups:
            message = ("no count vector satisfying every group quota reaches "
                       "the requested tolerance; the closest quota-feasible "
                       "witnesses below and above the target are provided, and "
                       "each group total is directly recomputable")
        return 200, {
            "status": "unsatisfiable",
            "within_tolerance": False,
            "best_absolute_error": result["best_distance"],
            "message": message,
            **common,
        }

    # Winners from the solver all attain one of the optimal masses; recover
    # the exact attained mass per vector (vectors are in canonical id order).
    optimal_set = set(result["optimal_masses"])
    explanations: list[dict[str, Any]] = []
    for vec in result["vectors"]:
        mass = sum(vec[pos] * solver.m[pos] for pos in range(solver.n))
        assert mass in optimal_set
        doc = _explanation(
            order, meta, mass, result["particle_count"], target, vec)
        if groups:
            doc["group_counts"] = [
                _group_count_doc(grp, order, vec) for grp in groups]
        explanations.append(doc)

    # A distinct witness proving non-uniqueness of the two-level optimum.
    alternative = None
    if not result["unique"] and len(explanations) >= 2:
        alternative = explanations[1]

    return 200, {
        "status": "optimal",
        "within_tolerance": True,
        "best_absolute_error": result["best_distance"],
        "optimal_total_masses": result["optimal_masses"],
        "particle_count": result["particle_count"],
        "num_optimal_explanations": result["num_optimal_explanations"],
        "unique": result["unique"],
        "alternative_witness": alternative,
        "truncated": result["truncated_list"],
        "explanations": explanations,
        **common,
    }
