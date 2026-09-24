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
          "components": [{"id": str, "mass": int, "min": int, "max": int}, ...]
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

    if errors:
        raise RequestError(errors)

    return {"target": target, "tolerance": tol, "components": components}
