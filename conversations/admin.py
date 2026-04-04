from django.contrib import admin
from .models import Session, Message, Profile


@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    """Individual chatbot conversation threads — tracks state, timing, and student linkage."""
    list_display = ('id', 'status', 'fsm_state', 'expires_at', 'created_at', 'updated_at')
    search_fields = ('id', 'fsm_state', 'status')
    list_filter = ('status',)


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    """Individual messages within conversations — content is encrypted for privacy."""
    list_display = ('id', 'session', 'role', 'fsm_state', 'created_at')
    search_fields = ('id', 'session__id', 'fsm_state', 'role')
    list_filter = ('role',)


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    """Extracted student profiles from conversations — grades, preferences, and traits discovered during chat."""
    list_display = ('id', 'session', 'version', 'created_at', 'updated_at')
    search_fields = ('id', 'session__id', 'version')
