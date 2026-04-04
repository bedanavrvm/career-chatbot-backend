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
# Base admin class with visible descriptions
# ============================================================
class DescriptiveModelAdmin(admin.ModelAdmin):
    """Base admin that shows the model's docstring as a description at the top of the changelist."""
    
    def changelist_view(self, request, extra_context=None):
        """Show the model description as an info message."""
        doc = self.__doc__
        if doc:
            # Strip leading/trailing whitespace for clean display
            description = doc.strip()
            self.message_user(request, description, level=messages.INFO)
        return super().changelist_view(request, extra_context)


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
class InstitutionAdmin(DescriptiveModelAdmin):
    """Universities and colleges — the institutions whose admission data the system tracks."""
    list_display = ("code", "name", "region", "county")
    search_fields = ("code", "name", "alias", "region", "county")
    list_filter = ("region", "county")


@admin.register(Field)
class FieldAdmin(DescriptiveModelAdmin):
    """Broad academic disciplines (e.g. Engineering, Medicine) that programmes are grouped under."""
    list_display = ("name", "parent")
    search_fields = ("name",)


@admin.register(Subject)
class SubjectAdmin(DescriptiveModelAdmin):
    """KCSE high school subjects — the grades students enter for eligibility checking."""
    list_display = ("code", "name", "group")
    search_fields = ("code", "name", "group")


@admin.register(Program)
class ProgramAdmin(DescriptiveModelAdmin):
    """University degree programmes — the core data students search and get recommended."""
    list_display = ("normalized_name", "institution", "level", "region", "subj1", "subj2", "subj3", "subj4")
    search_fields = ("normalized_name", "name", "code", "region", "campus")
    list_filter = ("level", "region", "institution")


@admin.register(YearlyCutoff)
class YearlyCutoffAdmin(DescriptiveModelAdmin):
    """Historical minimum cluster scores per programme per year — shows admission trends over time."""
    list_display = ("program", "year", "cutoff", "capacity")
    list_filter = ("year",)
    search_fields = ("program__normalized_name",)


@admin.register(InstitutionCampus)
class InstitutionCampusAdmin(DescriptiveModelAdmin):
    """Physical campus locations for each institution — many universities have multiple campuses."""
    list_display = ("institution", "campus", "region", "county", "town")
    search_fields = ("institution__code", "institution__name", "campus", "region", "county", "town")
    list_filter = ("region", "county")
