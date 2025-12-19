from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', views.login, name='login'),
    path('me/', views.me, name='me'),
    # Onboarding
    path('onboarding/me/', views.onboarding_me, name='onboarding_me'),
    path('onboarding/save/', views.onboarding_save, name='onboarding_save'),
    path('onboarding/dashboard/', views.onboarding_dashboard, name='onboarding_dashboard'),
]
