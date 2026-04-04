from django.contrib import admin, messages
from django.http import HttpResponse
from django.conf import settings
import csv
from pathlib import Path
from .models import (
    Institution,
    Field,
    Subject,
    Program,
    YearlyCutoff,
    NormalizationRule,
    DedupMatch,
    InstitutionCampus,
    ProgramOfferingAggregate,
    ProgramOfferingBroadAggregate,
    DedupCandidateGroup,
    DedupSummary,
    CodeCorrectionAudit,
    ETLRun,
    DQReportEntry,
    ClusterSubjectRule,
    ProgramRequirementNormalized,
    ProgramRequirementGroup,
    ProgramRequirementOption,
    CourseSuffixMapping,
)


# ============================================================
# HIDDEN FROM ADMIN (internal/ETL models — no demo data)
# ============================================================
# The following models are internal ETL artifacts, audit logs,
# or intermediate processing tables. They are intentionally
# hidden from the admin demo to avoid confusion.
#
# Hidden models:
#   - NormalizationRule, DedupMatch, DedupCandidateGroup,
#     DedupSummary, CodeCorrectionAudit
#   - ProgramOfferingAggregate, ProgramOfferingBroadAggregate
#   - ETLRun, DQReportEntry, ClusterSubjectRule
#   - ProgramRequirementNormalized, ProgramRequirementGroup,
#     ProgramRequirementOption, CourseSuffixMapping
# ============================================================


@admin.register(Institution)
class InstitutionAdmin(admin.ModelAdmin):
    """Universities and colleges — the institutions whose admission data the system tracks."""
    list_display = ("code", "name", "region", "county")
    search_fields = ("code", "name", "alias", "region", "county")
    list_filter = ("region", "county")


@admin.register(Field)
class FieldAdmin(admin.ModelAdmin):
    """Broad academic disciplines (e.g. Engineering, Medicine) that programmes are grouped under."""
    list_display = ("name", "parent")
    search_fields = ("name",)


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    """KCSE high school subjects — the grades students enter for eligibility checking."""
    list_display = ("code", "name", "group")
    search_fields = ("code", "name", "group")


@admin.register(Program)
class ProgramAdmin(admin.ModelAdmin):
    """University degree programmes — the core data students search and get recommended."""
    list_display = ("normalized_name", "institution", "level", "region", "subj1", "subj2", "subj3", "subj4")
    search_fields = ("normalized_name", "name", "code", "region", "campus")
    list_filter = ("level", "region", "institution")


@admin.register(YearlyCutoff)
class YearlyCutoffAdmin(admin.ModelAdmin):
    """Historical minimum cluster scores per programme per year — shows admission trends over time."""
    list_display = ("program", "year", "cutoff", "capacity")
    list_filter = ("year",)
    search_fields = ("program__normalized_name",)


@admin.register(InstitutionCampus)
class InstitutionCampusAdmin(admin.ModelAdmin):
    """Physical campus locations for each institution — many universities have multiple campuses."""
    list_display = ("institution", "campus", "region", "county", "town")
    search_fields = ("institution__code", "institution__name", "campus", "region", "county", "town")
    list_filter = ("region", "county")
