from django.contrib import admin

from .models import (
    OnetContentElement,
    OnetInterest,
    OnetOccupation,
    OnetRelatedOccupation,
    OnetScale,
    OnetSkill,
    OnetTaskStatement,
)

# Ensure mapping models are registered
from . import mapping_admin  # noqa: F401


@admin.register(OnetOccupation)
class OnetOccupationAdmin(admin.ModelAdmin):
    search_fields = ('onetsoc_code', 'title')
    list_display = ('onetsoc_code', 'title')


@admin.register(OnetContentElement)
class OnetContentElementAdmin(admin.ModelAdmin):
    search_fields = ('element_id', 'element_name')
    list_display = ('element_id', 'element_name')


@admin.register(OnetScale)
class OnetScaleAdmin(admin.ModelAdmin):
    search_fields = ('scale_id', 'scale_name')
    list_display = ('scale_id', 'scale_name')


@admin.register(OnetInterest)
class OnetInterestAdmin(admin.ModelAdmin):
    search_fields = ('onetsoc_code__onetsoc_code', 'element_id__element_id', 'scale_id__scale_id')
    list_display = ('onetsoc_code', 'element_id', 'scale_id', 'data_value', 'date_updated')


@admin.register(OnetSkill)
class OnetSkillAdmin(admin.ModelAdmin):
    search_fields = ('onetsoc_code__onetsoc_code', 'element_id__element_id', 'scale_id__scale_id')
    list_display = ('onetsoc_code', 'element_id', 'scale_id', 'data_value', 'date_updated')


@admin.register(OnetTaskStatement)
class OnetTaskStatementAdmin(admin.ModelAdmin):
    search_fields = ('onetsoc_code__onetsoc_code', 'task')
    list_display = ('task_id', 'onetsoc_code', 'task_type', 'date_updated')


@admin.register(OnetRelatedOccupation)
class OnetRelatedOccupationAdmin(admin.ModelAdmin):
    search_fields = ('onetsoc_code__onetsoc_code', 'related_onetsoc_code__onetsoc_code', 'relatedness_tier')
    list_display = ('onetsoc_code', 'related_onetsoc_code', 'relatedness_tier', 'related_index')
