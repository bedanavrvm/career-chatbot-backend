from django.urls import path
from .views import get_session, post_message, post_profile, delete_session, sessions_collection, session_recommendations

urlpatterns = [
    path('conversations/sessions', sessions_collection, name='conv_sessions_collection'),
    path('conversations/sessions/<uuid:session_id>', get_session, name='conv_get_session'),
    path('conversations/sessions/<uuid:session_id>/recommendations', session_recommendations, name='conv_session_recommendations'),
    path('conversations/sessions/<uuid:session_id>/messages', post_message, name='conv_post_message'),
    path('conversations/sessions/<uuid:session_id>/delete', delete_session, name='conv_delete_session'),
    path('conversations/profile', post_profile, name='conv_post_profile'),
]
