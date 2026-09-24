"""Structured, locatable validation errors.

Errors carry a JSON-Pointer style ``path`` (e.g. ``/components/2/mass``) so
that an analyst can pinpoint the offending field of a request.
"""

from __future__ import annotations

from typing import Any


class ValidationError(Exception):
    def __init__(self, code: str, message: str, path: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path:
            d["path"] = self.path
        return d


class RequestError(Exception):
    """Container for one or more field-level validation errors."""

    def __init__(self, errors: list[ValidationError]):
        self.errors = errors
        super().__init__("; ".join(f"{e.path or '/'}: {e.message}" for e in errors))

    def to_dict(self) -> dict[str, Any]:
        return {"errors": [e.to_dict() for e in self.errors]}
