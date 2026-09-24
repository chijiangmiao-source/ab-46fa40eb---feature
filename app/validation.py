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
MAX_GROUP_MEMBERS = 16


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


def parse_request(body: Any) -> dict[str, Any]:
    """Validate the inversion request body.

    Returns a normalized dict::

        {
          "target": int, "tolerance": int,
          "components": [{"id": str, "mass": int, "min": int, "max": int}, ...],
          "groups": [{"name": str|None, "members": [str, ...],
                      "min": int, "max": int}, ...]
        }

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

    # Bounds of every successfully identified component (duplicates excluded).
    ranges_by_id: dict[str, tuple[int, int]] = {}
    for raw in comps_raw:
        if isinstance(raw, dict) and isinstance(raw.get("id"), str):
            cid = raw["id"]
            if cid in seen_ids and cid not in ranges_by_id:
                lo_v = raw.get("min", 0)
                hi_v = raw.get("max", MAX_BOUND)
                if isinstance(lo_v, int) and not isinstance(lo_v, bool) \
                        and isinstance(hi_v, int) and not isinstance(hi_v, bool) \
                        and 0 <= lo_v <= hi_v <= MAX_BOUND:
                    ranges_by_id[cid] = (lo_v, hi_v)

    groups = _parse_groups(body.get("groups"), seen_ids, ranges_by_id, errors)

    if errors:
        raise RequestError(errors)

    return {"target": target, "tolerance": tol,
            "components": components, "groups": groups}


def _parse_groups(raw: Any, known_ids: set[str],
                  ranges_by_id: dict[str, tuple[int, int]],
                  errors: list[ValidationError]) -> list[dict[str, Any]]:
    """Validate the optional disjoint ``groups`` quota declaration."""
    if raw is None:
        return []
    gpath = "/groups"
    if not isinstance(raw, list):
        _err(errors, "invalid_type",
             "'groups' must be a list of group quota objects", gpath)
        return []
    if len(raw) > MAX_GROUPS:
        _err(errors, "out_of_range",
             f"'groups' may contain at most {MAX_GROUPS} groups, got {len(raw)}",
             gpath)

    groups: list[dict[str, Any]] = []
    owner: dict[str, int] = {}  # component id -> index of the group using it
    for gi, grp in enumerate(raw[:MAX_GROUPS]):
        path = f"{gpath}/{gi}"
        if not isinstance(grp, dict):
            _err(errors, "invalid_type", "group must be an object", path)
            continue

        name = grp.get("name", f"group-{gi}")
        if not isinstance(name, str) or not name.strip():
            _err(errors, "invalid_type",
                 "'name' must be a non-empty string", f"{path}/name")
            name = f"group-{gi}"

        members_raw = grp.get("members")
        if not isinstance(members_raw, list) or not members_raw:
            _err(errors, "invalid_type",
                 "'members' must be a non-empty list of submitted component ids",
                 f"{path}/members")
            members: list[str] = []
        elif len(members_raw) > MAX_GROUP_MEMBERS:
            _err(errors, "out_of_range",
                 f"'members' may reference at most {MAX_GROUP_MEMBERS} "
                 f"components, got {len(members_raw)}", f"{path}/members")
            members = []
        else:
            members = []
            local_seen: set[str] = set()
            for j, mid in enumerate(members_raw):
                mpath = f"{path}/members/{j}"
                if not isinstance(mid, str) or not mid.strip():
                    _err(errors, "invalid_type",
                         "group member must be a non-empty component id string",
                         mpath)
                    continue
                if mid not in known_ids:
                    _err(errors, "unknown_member",
                          f"group member {mid!r} is not a submitted component id",
                          mpath)
                    continue
                if mid in local_seen:
                    _err(errors, "duplicate_member",
                          f"component {mid!r} is listed more than once in group "
                          f"{name!r}", mpath)
                    continue
                local_seen.add(mid)
                if mid in owner:
                    _err(errors, "overlapping_group",
                          f"component {mid!r} already belongs to group "
                          f"index {owner[mid]}; groups must be pairwise disjoint",
                          f"{path}/members/{j}")
                    continue
                members.append(mid)
            for mid in members:
                owner.setdefault(mid, gi)

        qlo = _coerce_int(grp.get("min"), f"{path}/min", errors,
                          field="groups[].min")
        if qlo is not None and qlo < 0:
            _err(errors, "out_of_range",
                 "group 'min' must be non-negative", f"{path}/min")
        qhi = _coerce_int(grp.get("max"), f"{path}/max", errors,
                          field="groups[].max")
        if qhi is not None and qhi < 0:
            _err(errors, "out_of_range",
                 "group 'max' must be non-negative", f"{path}/max")
        if qlo is not None and qhi is not None and qlo > qhi:
            _err(errors, "inverted_bounds",
                 f"group 'min' ({qlo}) must not exceed 'max' ({qhi})",
                 f"{path}/min")

        # Reachability of the quota against the members' own count bounds.
        if members and qlo is not None and qhi is not None and qlo <= qhi:
            known = [ranges_by_id[mid] for mid in members
                     if mid in ranges_by_id]
            if len(known) == len(members):
                lo_sum = sum(lo for lo, _ in known)
                hi_sum = sum(hi for _, hi in known)
                if qhi < lo_sum or qlo > hi_sum:
                    _err(errors, "quota_unreachable",
                          f"group {name!r} quota [{qlo}, {qhi}] is unreachable: "
                          f"member bounds only allow total counts in "
                          f"[{lo_sum}, {hi_sum}]", f"{path}/min")

        groups.append({"name": name, "members": members,
                       "min": qlo if qlo is not None else 0,
                       "max": qhi if qhi is not None else 0})

    return groups
