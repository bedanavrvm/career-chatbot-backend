from django.contrib import admin

from .models import (
    OnetContentElement,
    OnetEducationTrainingExperience,
    OnetEteCategory,
    OnetJobZone,
    OnetJobZoneReference,
    OnetInterest,
    OnetOccupation,
    OnetOccupationSnapshot,
    OnetRelatedOccupation,
    OnetScale,
    OnetSkill,
    OnetTaskStatement,
)

# Ensure mapping models are registered
from . import mapping_admin  # noqa: F401


# ============================================================
# HIDDEN FROM ADMIN (internal O*NET structure — too granular)
# ============================================================
# The following models are O*NET internal scaffolding. They are
# kept in code but intentionally not exposed in the admin demo.
#
# Hidden models:
#   - OnetOccupationSnapshot, OnetContentElement, OnetScale,
#     OnetTaskStatement, OnetRelatedOccupation,
#     OnetJobZoneReference, OnetJobZone, OnetEteCategory,
#     OnetEducationTrainingExperience
# ============================================================


@admin.register(OnetOccupation)
class OnetOccupationAdmin(admin.ModelAdmin):
    """Master list of occupations from the O*NET database — the careers the system can recommend."""
    search_fields = ('onetsoc_code', 'title')
    list_display = ('onetsoc_code', 'title')


@admin.register(OnetInterest)
class OnetInterestAdmin(admin.ModelAdmin):
    """RIASEC interest scores per occupation — maps student personality types to suitable careers."""
    search_fields = ('onetsoc_code__onetsoc_code', 'element_id__element_id', 'scale_id__scale_id')
    list_display = ('onetsoc_code', 'element_id', 'scale_id', 'data_value', 'date_updated')


@admin.register(OnetSkill)
class OnetSkillAdmin(admin.ModelAdmin):
    """Skills required for each occupation — used to show students what they need to develop."""
    search_fields = ('onetsoc_code__onetsoc_code', 'element_id__element_id', 'scale_id__scale_id')
    list_display = ('onetsoc_code', 'element_id', 'scale_id', 'data_value', 'date_updated')
