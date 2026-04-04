from django.contrib import admin, messages
from .models import Session, Message, Profile


# ============================================================
# Base admin class with visible descriptions
# ============================================================
class DescriptiveModelAdmin(admin.ModelAdmin):
    """Base admin that shows the model's docstring as a description at the top of the changelist."""
    
    def changelist_view(self, request, extra_context=None):
        """Show the model description as an info message."""
        doc = self.__doc__
        if doc:
            description = doc.strip()
            self.message_user(request, description, level=messages.INFO)
        return super().changelist_view(request, extra_context)


@admin.register(Session)
class SessionAdmin(DescriptiveModelAdmin):
    """Individual chatbot conversation threads — tracks state, timing, and student linkage."""
    list_display = ('id', 'status', 'fsm_state', 'expires_at', 'created_at', 'updated_at')
    search_fields = ('id', 'fsm_state', 'status')
    list_filter = ('status',)


@admin.register(Message)
class MessageAdmin(DescriptiveModelAdmin):
    """Individual messages within conversations — content is encrypted for privacy."""
    list_display = ('id', 'session', 'role', 'fsm_state', 'created_at')
    search_fields = ('id', 'session__id', 'fsm_state', 'role')
    list_filter = ('role',)


@admin.register(Profile)
class ProfileAdmin(DescriptiveModelAdmin):
    """Extracted student profiles from conversations — grades, preferences, and traits discovered during chat."""
    list_display = ('id', 'session', 'version', 'created_at', 'updated_at')
    search_fields = ('id', 'session__id', 'version')
