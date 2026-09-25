"""Request parsing and exact-integer validation.

Every numeric quantity (masses, target, tolerance, bounds) must be supplied as
an exact integer (or an *integral* JSON number -- JSON has no integer type, so
``12.0`` is accepted but ``12.5`` is rejected).  Internally everything is a
Python ``int``; no floating point is ever involved in the computation.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import RequestError, ValidationError

MAX_BOUND = 1_000_000
MIN_COMPONENTS = 2
MAX_COMPONENTS = 16
MAX_GROUPS = 16


def _err(errors: list[ValidationError], code: str, message: str, path: str) -> None:
    errors.append(ValidationError(code, message, path))


def _coerce_int(value: Any, path: str, errors: list[ValidationError], *,
                field: str) -> int | None:
    """Return value as an exact int, or record a validation error.

    Booleans are rejected (``True`` is not a mass); floats are accepted only
    when they are integral (e.g. ``100.0``), never with a fraction.
    """
    if isinstance(value, bool):
        _err(errors, "invalid_type", f"'{field}' must be an integer, got boolean", path)
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and float(value).is_integer():
            return int(value)
        _err(errors, "invalid_integer",
             f"'{field}' must be an exact integer value", path)
        return None
    _err(errors, "invalid_type",
         f"'{field}' must be an integer, got {type(value).__name__}", path)
    return None


def parse_request(body: Any, *, accept_groups: bool = False) -> dict[str, Any]:
    """Validate the inversion request body.

    Returns a normalized dict::

        {
          "target": int, "tolerance": int,
          "components": [{"id": str, "mass": int, "min": int, "max": int}, ...],
          "groups": [...] | None
        }

    The optional ``groups`` declaration is parsed only on entry points that
    accept constrained reviews (``accept_groups``); the original inversion
    interface ignores the field entirely.

    Raises :class:`RequestError` with locatable field errors otherwise.
    """
    errors: list[ValidationError] = []

    if not isinstance(body, dict):
        raise RequestError([ValidationError(
            "invalid_type", "request body must be a JSON object", "")])

    target = _coerce_int(body.get("target"), "/target", errors, field="target")
    if target is not None and target < 0:
        _err(errors, "out_of_range", "'target' mass must be non-negative", "/target")

    tol = _coerce_int(body.get("tolerance"), "/tolerance", errors, field="tolerance")
    if tol is not None and tol < 0:
        _err(errors, "out_of_range", "'tolerance' must be non-negative", "/tolerance")

    comps_raw = body.get("components")
    if not isinstance(comps_raw, list):
        _err(errors, "invalid_type",
             "'components' must be a list of 2..16 component objects",
             "/components")
        comps_raw = []
    elif not (MIN_COMPONENTS <= len(comps_raw) <= MAX_COMPONENTS):
        _err(errors, "out_of_range",
             f"'components' must contain between {MIN_COMPONENTS} and "
             f"{MAX_COMPONENTS} unique components, got {len(comps_raw)}",
             "/components")

    components: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(comps_raw):
        path = f"/components/{i}"
        if not isinstance(raw, dict):
            _err(errors, "invalid_type", "component must be an object", path)
            continue

        cid = raw.get("id")
        if not isinstance(cid, str) or not cid.strip():
            _err(errors, "invalid_type",
                 "'id' must be a non-empty string", f"{path}/id")
            cid = None
        elif cid in seen_ids:
            _err(errors, "duplicate_id",
                 f"duplicate component id {cid!r}", f"{path}/id")
        else:
            seen_ids.add(cid)

        mass = _coerce_int(raw.get("mass"), f"{path}/mass", errors, field="mass")
        if mass is not None and mass <= 0:
            _err(errors, "out_of_range",
                 "'mass' must be a positive integer (micro-daltons)", f"{path}/mass")

        lo = _coerce_int(raw.get("min", 0), f"{path}/min", errors, field="min")
        if lo is not None and not (0 <= lo <= MAX_BOUND):
            _err(errors, "out_of_range",
                 f"'min' must be between 0 and {MAX_BOUND}", f"{path}/min")

        hi = _coerce_int(raw.get("max", MAX_BOUND), f"{path}/max", errors, field="max")
        if hi is not None and not (0 <= hi <= MAX_BOUND):
            _err(errors, "out_of_range",
                 f"'max' must be between 0 and {MAX_BOUND}", f"{path}/max")

        if lo is not None and hi is not None and lo > hi:
            _err(errors, "inverted_bounds",
                 f"'min' ({lo}) must not exceed 'max' ({hi})", f"{path}/min")

        components.append({
            "id": cid if cid is not None else f"__invalid_{i}",
            "mass": mass if mass is not None and mass > 0 else 1,
            "min": lo if lo is not None else 0,
            "max": hi if hi is not None else 0,
        })

    # Component-group count quotas are validated only after the component list
    # itself has been validated, so every referenced identifier is known and
    # every member bound is trustworthy.
    groups = None
    if accept_groups:
        valid_components = [c for c in components
                            if not c["id"].startswith("__invalid_")]
        meta = {c["id"]: c for c in valid_components}
        # Only trust member min/max sums for reachability checks when the
        # component fields themselves validated cleanly.
        bounds_trustworthy = not any(
            e.path.startswith("/components/")
            and (e.path.endswith("/min") or e.path.endswith("/max"))
            for e in errors)
        groups = _parse_groups(body.get("groups"), set(meta), meta, errors,
                               check_reachable=bounds_trustworthy)

    if errors:
        raise RequestError(errors)

    return {"target": target, "tolerance": tol, "components": components,
            "groups": groups}


def _parse_groups(raw: Any, known_ids: set[str],
                  meta: dict[str, dict[str, Any]],
                  errors: list[ValidationError], *,
                  check_reachable: bool = True) -> list[dict[str, Any]] | None:
    """Validate the optional ``groups`` declaration.

    Each group is ``{"name"?: str, "components": [id, ...], "min": int,
    "max": int}`` declaring a closed interval for the *total particle count*
    over its members.  Groups must each reference at least one submitted
    component, be pairwise disjoint, and their quotas must be attainable given
    the member-level ``min``/``max`` bounds.

    Returns the normalized list (possibly empty), or ``None`` when the field
    was absent.  All problems are appended to ``errors``.
    """
    if raw is None:
        return None

    path0 = "/groups"
    if not isinstance(raw, list):
        _err(errors, "invalid_type",
             "'groups' must be a list of group objects", path0)
        return None
    if len(raw) > MAX_GROUPS:
        _err(errors, "out_of_range",
             f"'groups' may contain at most {MAX_GROUPS} groups, got "
             f"{len(raw)}", path0)

    groups: list[dict[str, Any]] = []
    names_seen: set[str] = set()
    ownership: dict[str, int] = {}

    for gi, grp in enumerate(raw):
        gpath = f"{path0}/{gi}"
        if not isinstance(grp, dict):
            _err(errors, "invalid_type", "group must be an object", gpath)
            continue

        gname = grp.get("name", f"group-{gi}")
        if not isinstance(gname, str) or not gname.strip():
            _err(errors, "invalid_type",
                 "group 'name' must be a non-empty string", f"{gpath}/name")
            gname = f"__invalid_name_{gi}"
        elif gname in names_seen:
            _err(errors, "duplicate_group_name",
                 f"duplicate group name {gname!r}", f"{gpath}/name")
        else:
            names_seen.add(gname)

        glo = _coerce_int(grp.get("min"), f"{gpath}/min", errors,
                          field="group min")
        if glo is not None and not (0 <= glo <= MAX_BOUND):
            _err(errors, "out_of_range",
                 f"group 'min' must be between 0 and {MAX_BOUND}",
                 f"{gpath}/min")

        ghi = _coerce_int(grp.get("max"), f"{gpath}/max", errors,
                          field="group max")
        if ghi is not None and not (0 <= ghi <= MAX_BOUND):
            _err(errors, "out_of_range",
                 f"group 'max' must be between 0 and {MAX_BOUND}",
                 f"{gpath}/max")

        if glo is not None and ghi is not None and glo > ghi:
            _err(errors, "inverted_bounds",
                 f"group 'min' ({glo}) must not exceed group 'max' ({ghi})",
                 f"{gpath}/min")

        members_raw = grp.get("components")
        members: list[str] = []
        if not isinstance(members_raw, list) or not members_raw:
            _err(errors, "invalid_type",
                 "group 'components' must be a non-empty list of submitted "
                 "component ids", f"{gpath}/components")
        else:
            local_seen: set[str] = set()
            for mi, ref in enumerate(members_raw):
                mpath = f"{gpath}/components/{mi}"
                if not isinstance(ref, str):
                    _err(errors, "invalid_type",
                         "group member must be a component id string", mpath)
                    continue
                if ref not in known_ids:
                    _err(errors, "unknown_component",
                         f"group references unknown component id {ref!r}; "
                         "declare it in 'components' first", mpath)
                    continue
                if ref in local_seen:
                    _err(errors, "duplicate_member",
                         f"component {ref!r} is listed more than once "
                         f"in group {gname!r}", mpath)
                    continue
                local_seen.add(ref)
                # Groups must be pairwise disjoint.
                if ref in ownership:
                    _err(errors, "overlapping_groups",
                         f"component {ref!r} belongs to multiple groups "
                         f"({groups[ownership[ref]]['name']!r} and "
                         f"{gname!r}); groups must be pairwise disjoint",
                         mpath)
                    continue
                ownership[ref] = gi
                members.append(ref)

        # Quota attainability against the member-level bounds: the smallest
        # possible group total is sum(min), the largest is sum(max).  Skipped
        # when the member bounds themselves failed validation.
        if (check_reachable and glo is not None and ghi is not None
                and members):
            sum_lo = sum(meta[cid]["min"] for cid in members)
            sum_hi = sum(meta[cid]["max"] for cid in members)
            if glo < sum_lo:
                _err(errors, "quota_unreachable",
                     f"group {gname!r} quota [{glo}, {ghi}] is unreachable: "
                     f"member-level minimums already total {sum_lo}",
                     f"{gpath}/min")
            if ghi > sum_hi:
                _err(errors, "quota_unreachable",
                     f"group {gname!r} quota [{glo}, {ghi}] is unreachable: "
                     f"member-level maximums only total {sum_hi}",
                     f"{gpath}/max")

        groups.append({
            "name": gname,
            "components": members,
            "min": glo if glo is not None else 0,
            "max": ghi if ghi is not None else 0,
        })

    return groups
