"""Application service: turn validated requests into response documents."""

from __future__ import annotations

import os
from typing import Any

from .errors import RequestError, ValidationError
from .solver import BudgetExceeded, Component, Solver
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


def _group_totals_doc(solver: Solver,
                      vec: list[int] | tuple[int, ...]) -> list[dict[str, Any]]:
    """Per-group totals directly recomputable from the reported counts."""
    docs = []
    for g, members in enumerate(solver.gmembers):
        total = sum(vec[p] for p in members)
        docs.append({
            "name": solver.gnames[g],
            "components": [solver.ids[p] for p in members],
            "total": total,
            "min": solver.glo[g],
            "max": solver.ghi[g],
            "within_quota": solver.glo[g] <= total <= solver.ghi[g],
        })
    return docs


def _explanation(order: list[str], meta: dict[str, dict[str, int]],
                 mass: int, particle_count: int, target: int,
                 vec: tuple[int, ...],
                 solver: Solver | None = None) -> dict[str, Any]:
    counts = _counts_doc(order, meta, vec)
    recomputed = sum(c["mass_contribution"] for c in counts)
    doc = {
        "total_mass": mass,
        "error": mass - target,                 # signed, in micro-daltons
        "absolute_error": abs(mass - target),
        "particle_count": particle_count,
        "counts": counts,
        "recomputed_total_mass": recomputed,    # equals total_mass exactly
    }
    if solver is not None and solver.G:
        doc["group_totals"] = _group_totals_doc(solver, vec)
    return doc


def _witness_doc(order: list[str], meta: dict[str, dict[str, int]],
                 witness: dict[str, Any] | None,
                 target: int) -> dict[str, Any] | None:
    if witness is None:
        return None
    return {
        "total_mass": witness["total_mass"],
        "error": witness["error"],
        "absolute_error": witness["absolute_error"],
        "particle_count": witness["particle_count"],
        "counts": _counts_doc(order, meta, witness["vector"]),
        "recomputed_total_mass": witness["total_mass"],
        "side": "below" if witness["error"] <= 0 else "above",
        **({"group_totals": witness["group_totals"]}
           if witness.get("group_totals") is not None else {}),
    }


def _groups_echo(solver: Solver) -> list[dict[str, Any]] | None:
    if not solver.G:
        return None
    return [
        {
            "name": solver.gnames[g],
            "components": [solver.ids[p] for p in solver.gmembers[g]],
            "min": solver.glo[g],
            "max": solver.ghi[g],
        }
        for g in range(solver.G)
    ]


def _run(body: Any, *, constrained: bool) -> tuple[int, dict[str, Any]]:
    parsed = parse_request(body, accept_groups=constrained)

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

    solver = Solver(components, parsed["target"], parsed["tolerance"],
                    groups=parsed.get("groups"),
                    node_budget=_node_budget())
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

    order = result["component_order"]
    target = result["target"]

    common = {
        "target": target,
        "tolerance": result["tolerance"],
        "mass_unit": "micro_dalton",
        "count_bounds_hard_max": MAX_BOUND,
        "component_order": order,
        "nearest_below": _witness_doc(order, meta, result["nearest_below"], target),
        "nearest_above": _witness_doc(order, meta, result["nearest_above"], target),
    }
    groups_echo = _groups_echo(solver)
    if groups_echo is not None:
        common["groups"] = groups_echo

    if result["status"] == "unsatisfiable":
        message = ("no reachable total mass lies within the requested "
                    "tolerance; the closest attainable witnesses below "
                    "and above the target are provided")
        if solver.G:
            message = ("no count vector satisfying every declared group "
                       "quota simultaneously reaches a total mass within "
                       "the requested tolerance; the closest quota-feasible "
                       "witnesses below and above the target are provided, "
                       "and each group total can be recomputed from counts")
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
        explanations.append(_explanation(
            order, meta, mass, result["particle_count"], target, vec,
            solver if solver.G else None))

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


def invert(body: Any) -> tuple[int, dict[str, Any]]:
    """Execute one unconstrained inversion. Returns (http_status, body)."""
    return _run(body, constrained=False)


def invert_constrained(body: Any) -> tuple[int, dict[str, Any]]:
    """Execute one inversion under declared disjoint group quotas."""
    return _run(body, constrained=True)
