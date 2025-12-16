from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.db import transaction
import os
import base64
import json
import firebase_admin
from firebase_admin import auth as fb_auth, credentials
from .models import UserProfile
from .serializers import UserProfileSerializer


# Ensure Firebase Admin is initialized (in case urls init didn't run yet)
if not firebase_admin._apps:
    b64 = os.getenv('FIREBASE_CREDENTIALS_JSON_B64')
    if b64:
        try:
            data = json.loads(base64.b64decode(b64).decode('utf-8'))
            cred = credentials.Certificate(data)
            firebase_admin.initialize_app(cred)
        except Exception:
            pass


def _get_token_from_request(request):
    auth_header = request.META.get('HTTP_AUTHORIZATION', '')
    if auth_header.startswith('Bearer '):
        return auth_header.split(' ', 1)[1]
    if request.method in ('POST', 'PUT', 'PATCH'):
        try:
            body = request.data
        except Exception:
            body = {}
        token = body.get('id_token')
        if token:
            return token
    token = request.GET.get('id_token')
    return token


def _upsert_profile_from_claims(claims):
    uid = claims.get('uid')
    email = claims.get('email')
    name = claims.get('name') or claims.get('displayName')
    picture = claims.get('picture')
    if not uid:
        return None
    with transaction.atomic():
        profile, _ = UserProfile.objects.select_for_update().get_or_create(uid=uid)
        # Update fields if changed
        changed = False
        for field, value in (('email', email), ('display_name', name), ('photo_url', picture)):
            if value and getattr(profile, field) != value:
                setattr(profile, field, value)
                changed = True
        if changed:
            profile.save()
    return profile


@api_view(['POST'])
def register(request):
    """Create a user profile if it doesn't exist; otherwise return existing."""
    token = _get_token_from_request(request)
    if not token:
        return Response({"detail": "Missing id_token"}, status=status.HTTP_400_BAD_REQUEST)
    try:
        claims = fb_auth.verify_id_token(token)
        profile = _upsert_profile_from_claims(claims)
        if not profile:
            return Response({"detail": "Invalid token"}, status=status.HTTP_401_UNAUTHORIZED)
        return Response(UserProfileSerializer(profile).data)
    except Exception:
        return Response({"detail": "Invalid token"}, status=status.HTTP_401_UNAUTHORIZED)


@api_view(['POST'])
def login(request):
    """Verify token and upsert profile (Firebase handles auth)."""
    return register(request)


@api_view(['GET'])
def me(request):
    token = _get_token_from_request(request)
    if not token:
        return Response({"detail": "Missing bearer token"}, status=status.HTTP_401_UNAUTHORIZED)
    try:
        claims = fb_auth.verify_id_token(token)
        profile = _upsert_profile_from_claims(claims)
        if not profile:
            return Response({"detail": "Invalid token"}, status=status.HTTP_401_UNAUTHORIZED)
        return Response(UserProfileSerializer(profile).data)
    except Exception:
        return Response({"detail": "Invalid token"}, status=status.HTTP_401_UNAUTHORIZED)
