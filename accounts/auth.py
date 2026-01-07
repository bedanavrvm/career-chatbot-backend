import base64
import json
import os
import sys
from typing import Any, Callable, Optional, Tuple

import firebase_admin
from django.conf import settings
from django.http import JsonResponse
from firebase_admin import auth as fb_auth
from firebase_admin import credentials


_FIREBASE_INIT_ERROR: str = ''


RUNNING_TESTS = 'test' in sys.argv


def firebase_init_error() -> str:
    return _FIREBASE_INIT_ERROR


def ensure_firebase_initialized() -> bool:
    global _FIREBASE_INIT_ERROR
    if RUNNING_TESTS:
        # Tests should not require real Firebase credentials.
        _FIREBASE_INIT_ERROR = ''
        return True
    if firebase_admin._apps:
        return True

    path = (os.getenv('FIREBASE_CREDENTIALS_JSON_PATH') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS') or '').strip()
    if path:
        try:
            cred = credentials.Certificate(path)
            firebase_admin.initialize_app(cred)
            _FIREBASE_INIT_ERROR = ''
            return True
        except Exception as e:
            _FIREBASE_INIT_ERROR = f"{e.__class__.__name__}: {str(e)}".strip()
            return False

    b64 = (os.getenv('FIREBASE_CREDENTIALS_JSON_B64') or '').strip()
    if not b64:
        _FIREBASE_INIT_ERROR = 'Missing FIREBASE_CREDENTIALS_JSON_B64'
        return False

    try:
        data = json.loads(base64.b64decode(b64).decode('utf-8'))
        cred = credentials.Certificate(data)
        firebase_admin.initialize_app(cred)
        _FIREBASE_INIT_ERROR = ''
        return True
    except Exception as e:
        _FIREBASE_INIT_ERROR = f"{e.__class__.__name__}: {str(e)}".strip()
        return False


def verify_firebase_id_token(token: str) -> tuple[Optional[dict], Optional[str], Optional[int]]:
    """Verify a Firebase ID token.

    Returns: (claims, error_detail, http_status)
    """
    tok = (token or '').strip()
    if not tok:
        return None, 'Missing bearer token', 401

    if RUNNING_TESTS:
        # Minimal deterministic claims for tests.
        return {
            'uid': 'test-user',
            'email': 'test@example.com',
            'name': 'Test User',
        }, None, None

    if not ensure_firebase_initialized():
        detail = 'Firebase admin not initialized'
        if _FIREBASE_INIT_ERROR:
            detail = f"{detail}: {_FIREBASE_INIT_ERROR}"
        return None, detail, 503

    try:
        claims = fb_auth.verify_id_token(tok)
        if not isinstance(claims, dict):
            return None, 'Invalid token', 401
        return claims, None, None
    except Exception:
        return None, 'Invalid token', 401


def get_bearer_token(request) -> str:
    auth_header = request.META.get('HTTP_AUTHORIZATION', '')
    if auth_header:
        parts = auth_header.split(' ', 1)
        if len(parts) == 2 and parts[0].strip().lower() == 'bearer':
            return parts[1].strip()

    if not getattr(settings, 'DEBUG', False):
        return ''

    if request.method in ('POST', 'PUT', 'PATCH'):
        body: Any = {}
        if hasattr(request, 'data'):
            try:
                body = request.data or {}
            except Exception:
                body = {}
        else:
            try:
                raw = request.body.decode('utf-8') if hasattr(request, 'body') else ''
                body = json.loads(raw or '{}') if raw else {}
            except Exception:
                body = {}
        if isinstance(body, dict):
            for key in ('id_token', 'token', 'access_token', 'accessToken'):
                token = (body.get(key) or '').strip()
                if token:
                    return token

    token = (request.GET.get('id_token') or request.GET.get('token') or '').strip()
    return token


def require_firebase_uid(
    request,
    *,
    response_factory: Optional[Callable[[dict, int], Any]] = None,
) -> Tuple[Optional[str], Any]:
    if response_factory is None:
        response_factory = lambda data, status: JsonResponse(data, status=status)

    token = get_bearer_token(request)
    claims, err, http_status = verify_firebase_id_token(token)
    if err:
        st = int(http_status or 401)
        code = 'unauthorized'
        if st == 503:
            code = 'firebase_unavailable'
        elif st == 401 and str(err) == 'Missing bearer token':
            code = 'missing_token'
        return None, response_factory({'detail': err, 'code': code}, st)

    uid = (claims or {}).get('uid') if isinstance(claims, dict) else None
    if not uid:
        return None, response_factory({'detail': 'Invalid token', 'code': 'invalid_token'}, 401)
    return str(uid), None
