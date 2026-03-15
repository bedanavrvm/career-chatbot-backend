from django.contrib import admin

from .mapping_models import OnetFieldOccupationMapping


@admin.register(OnetFieldOccupationMapping)
class OnetFieldOccupationMappingAdmin(admin.ModelAdmin):
    search_fields = ('field__name', 'field__slug', 'occupation_code')
    list_display = ('field', 'occupation_code', 'weight', 'notes', 'updated_at')
    list_filter = ('field',)
