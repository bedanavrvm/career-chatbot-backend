from django.urls import path
from .views import get_session, post_message, post_profile

urlpatterns = [
    path('conversations/sessions/<uuid:session_id>', get_session, name='conv_get_session'),
    path('conversations/sessions/<uuid:session_id>/messages', post_message, name='conv_post_message'),
    path('conversations/profile', post_profile, name='conv_post_profile'),
]
