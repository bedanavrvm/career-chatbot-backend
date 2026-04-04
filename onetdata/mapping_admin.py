from django.contrib import admin, messages

from .mapping_models import OnetFieldOccupationMapping


class DescriptiveModelAdmin(admin.ModelAdmin):
    """Base admin that shows the model's docstring as a description at the top of the changelist."""
    
    def changelist_view(self, request, extra_context=None):
        """Show the model description as an info message."""
        doc = self.__doc__
        if doc:
            description = doc.strip()
            self.message_user(request, description, level=messages.INFO)
        return super().changelist_view(request, extra_context)


@admin.register(OnetFieldOccupationMapping)
class OnetFieldOccupationMappingAdmin(DescriptiveModelAdmin):
    """Bridge between academic fields and O*NET careers — tells the system which careers each programme leads to."""
    search_fields = ('field__name', 'field__slug', 'occupation_code')
    list_display = ('field', 'occupation_code', 'weight', 'notes', 'updated_at')
    list_filter = ('field',)
