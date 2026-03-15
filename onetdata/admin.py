from django.contrib import admin

from .models import (
    OnetContentElement,
    OnetEducationTrainingExperience,
    OnetEteCategory,
    OnetJobZone,
    OnetJobZoneReference,
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


@admin.register(OnetJobZoneReference)
class OnetJobZoneReferenceAdmin(admin.ModelAdmin):
    search_fields = ('job_zone', 'name')
    list_display = ('job_zone', 'name', 'svp_range')


@admin.register(OnetJobZone)
class OnetJobZoneAdmin(admin.ModelAdmin):
    search_fields = ('onetsoc_code__onetsoc_code',)
    list_display = ('onetsoc_code', 'job_zone', 'date_updated', 'domain_source')
    list_filter = ('job_zone',)


@admin.register(OnetEteCategory)
class OnetEteCategoryAdmin(admin.ModelAdmin):
    search_fields = ('element_id__element_id', 'scale_id__scale_id')
    list_display = ('element_id', 'scale_id', 'category')
    list_filter = ('element_id', 'scale_id')


@admin.register(OnetEducationTrainingExperience)
class OnetEducationTrainingExperienceAdmin(admin.ModelAdmin):
    search_fields = ('onetsoc_code__onetsoc_code', 'element_id__element_id', 'scale_id__scale_id')
    list_display = ('onetsoc_code', 'element_id', 'scale_id', 'category', 'data_value', 'date_updated')
    list_filter = ('element_id', 'scale_id')
