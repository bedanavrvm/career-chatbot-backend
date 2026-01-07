from __future__ import annotations

from typing import Any

from rest_framework.response import Response


def error_response(
    detail: str,
    *,
    status_code: int,
    code: str | None = None,
    fields: dict[str, Any] | None = None,
) -> Response:
    payload: dict[str, Any] = {"detail": detail}
    if code:
        payload["code"] = code
    if fields:
        payload["fields"] = fields
    return Response(payload, status=int(status_code))
