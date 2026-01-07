from __future__ import annotations

from typing import Any

from rest_framework import status
from rest_framework.exceptions import APIException
from rest_framework.response import Response
from rest_framework.views import exception_handler


def drf_exception_handler(exc: Exception, context: dict[str, Any]) -> Response | None:
    """Global DRF exception handler.

    Normalizes errors to:
      {"detail": str, "code"?: str, "fields"?: dict}

    - ValidationError => fields populated
    - APIException => code populated
    """

    resp = exception_handler(exc, context)
    if resp is None:
        return None

    data: dict[str, Any] = {}

    try:
        raw = resp.data
    except Exception:
        raw = None

    if isinstance(raw, dict):
        if "detail" in raw and isinstance(raw.get("detail"), (str, list, dict)):
            data["detail"] = raw.get("detail")
        elif "message" in raw:
            data["detail"] = raw.get("message")
        else:
            # Treat as validation-style dict of fields.
            data["detail"] = "Invalid request"
            data["fields"] = raw

        fields = raw.get("fields") if isinstance(raw.get("fields"), dict) else None
        if fields:
            data["fields"] = fields

    elif isinstance(raw, list):
        data["detail"] = "Invalid request"
        data["fields"] = {"non_field_errors": raw}
    elif raw is None:
        data["detail"] = "Request failed"
    else:
        data["detail"] = str(raw)

    if isinstance(exc, APIException):
        try:
            code = exc.get_codes()
        except Exception:
            code = None
        if isinstance(code, str):
            data["code"] = code
        elif isinstance(code, dict):
            data["code"] = "validation_error"
        elif isinstance(code, list):
            data["code"] = "validation_error"
        else:
            data["code"] = getattr(exc, "default_code", "error")

    # Ensure detail is string for common cases
    if isinstance(data.get("detail"), list):
        data["detail"] = "Invalid request"

    resp.data = data

    # Ensure 500s have a consistent envelope
    if int(getattr(resp, "status_code", 0) or 0) >= 500 and not data.get("detail"):
        resp.data = {"detail": "Server error", "code": "server_error"}

    # Ensure status code exists
    if not getattr(resp, "status_code", None):
        resp.status_code = status.HTTP_400_BAD_REQUEST

    return resp
