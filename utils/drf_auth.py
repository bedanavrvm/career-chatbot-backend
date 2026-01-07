from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import APIException, AuthenticationFailed
from rest_framework.permissions import BasePermission

from accounts.auth import get_bearer_token, verify_firebase_id_token


class FirebaseUnavailable(APIException):
    status_code = 503
    default_detail = 'Firebase admin not initialized'
    default_code = 'firebase_unavailable'


@dataclass(frozen=True)
class FirebaseUser:
    uid: str
    claims: dict

    @property
    def is_authenticated(self) -> bool:
        return True


class FirebaseAuthentication(BaseAuthentication):
    def authenticate(self, request) -> Optional[Tuple[FirebaseUser, dict]]:
        token = get_bearer_token(request)
        claims, err, http_status = verify_firebase_id_token(token)
        if err:
            if int(http_status or 0) == 503:
                raise FirebaseUnavailable(err)
            raise AuthenticationFailed(err)

        uid = (claims or {}).get('uid') if isinstance(claims, dict) else None
        if not uid:
            raise AuthenticationFailed('Invalid token')

        user = FirebaseUser(uid=str(uid), claims=claims)
        return user, claims


class IsFirebaseAuthenticated(BasePermission):
    def has_permission(self, request, view) -> bool:
        user = getattr(request, 'user', None)
        return bool(user and getattr(user, 'is_authenticated', False))


def optional_firebase_uid(request) -> Optional[str]:
    token = get_bearer_token(request)
    if not token:
        return None
    claims, err, _http_status = verify_firebase_id_token(token)
    if err:
        return None
    uid = (claims or {}).get('uid') if isinstance(claims, dict) else None
    return str(uid) if uid else None
