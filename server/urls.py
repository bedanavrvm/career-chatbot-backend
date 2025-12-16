"""
URL configuration for server project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse
import os
import base64
import json
import firebase_admin
from firebase_admin import auth as fb_auth, credentials

if not firebase_admin._apps:
    b64 = os.getenv('FIREBASE_CREDENTIALS_JSON_B64')
    if b64:
        try:
            data = json.loads(base64.b64decode(b64).decode('utf-8'))
            cred = credentials.Certificate(data)
            firebase_admin.initialize_app(cred)
        except Exception:
            # Fallback: app remains uninitialized; secure endpoints will error until configured
            pass


def health(_request):
    return JsonResponse({"status": "ok"})


def secure_ping(request):
    auth_header = request.META.get('HTTP_AUTHORIZATION', '')
    token = ''
    if auth_header.startswith('Bearer '):
        token = auth_header.split(' ', 1)[1]
    elif 'token' in request.GET:
        token = request.GET.get('token', '')

    if not token:
        return JsonResponse({"detail": "Missing bearer token"}, status=401)

    try:
        decoded = fb_auth.verify_id_token(token)
        uid = decoded.get('uid')
        return JsonResponse({"status": "ok", "uid": uid})
    except Exception:
        return JsonResponse({"detail": "Invalid token"}, status=401)


urlpatterns = [
    path('_health/', health, name='health'),
    path('_health', health, name='health_no_slash'),
    path('api/auth/', include('accounts.urls')),
    path('api/secure-ping/', secure_ping, name='secure_ping'),
    path('admin/', admin.site.urls),
]
