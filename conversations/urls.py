from django.urls import path
from .views import (
    get_session, post_message, post_profile, delete_session,
    sessions_collection, session_recommendations,
    post_message_async, get_task_status,
    post_message_stream,
)

urlpatterns = [
    path('conversations/sessions', sessions_collection, name='conv_sessions_collection'),
    path('conversations/sessions/<uuid:session_id>', get_session, name='conv_get_session'),
    path('conversations/sessions/<uuid:session_id>/recommendations', session_recommendations, name='conv_session_recommendations'),
    path('conversations/sessions/<uuid:session_id>/messages', post_message, name='conv_post_message'),
    # Async (non-blocking) variant — dispatches to Celery, returns task_id
    path('conversations/sessions/<uuid:session_id>/messages/async', post_message_async, name='conv_post_message_async'),
    # SSE streaming variant — streams reply tokens as they arrive from Gemini
    path('conversations/sessions/<uuid:session_id>/messages/stream', post_message_stream, name='conv_post_message_stream'),
    path('conversations/sessions/<uuid:session_id>/delete', delete_session, name='conv_delete_session'),
    path('conversations/profile', post_profile, name='conv_post_profile'),
    # Celery task status polling
    path('conversations/tasks/<str:task_id>/status', get_task_status, name='conv_task_status'),
]

