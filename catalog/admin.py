from django.contrib import admin
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


@admin.register(Institution)
class InstitutionAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "region", "county")
    search_fields = ("code", "name", "alias", "region", "county")
    list_filter = ("region", "county")


@admin.register(Field)
class FieldAdmin(admin.ModelAdmin):
    list_display = ("name", "parent")
    search_fields = ("name",)


@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "group")
    search_fields = ("code", "name", "group")


@admin.register(Program)
class ProgramAdmin(admin.ModelAdmin):
    list_display = ("normalized_name", "institution", "level", "region", "subj1", "subj2", "subj3", "subj4")
    search_fields = ("normalized_name", "name", "code", "region", "campus")
    list_filter = ("level", "region", "institution")


@admin.register(YearlyCutoff)
class YearlyCutoffAdmin(admin.ModelAdmin):
    list_display = ("program", "year", "cutoff", "capacity")
    list_filter = ("year",)
    search_fields = ("program__normalized_name",)


@admin.register(NormalizationRule)
class NormalizationRuleAdmin(admin.ModelAdmin):
    list_display = ("type", "source_value", "normalized_value")
    search_fields = ("type", "source_value", "normalized_value")
    list_filter = ("type",)


@admin.register(DedupMatch)
class DedupMatchAdmin(admin.ModelAdmin):
    list_display = ("master_program", "duplicate_program", "reason", "created_at")
    search_fields = ("master_program__normalized_name", "duplicate_program__normalized_name")


@admin.register(InstitutionCampus)
class InstitutionCampusAdmin(admin.ModelAdmin):
    list_display = ("institution", "campus", "region", "county", "town")
    search_fields = ("institution__code", "institution__name", "campus", "region", "county", "town")
    list_filter = ("region", "county")


@admin.register(ProgramOfferingAggregate)
class ProgramOfferingAggregateAdmin(admin.ModelAdmin):
    list_display = ("program_normalized_name", "course_suffix", "offerings_count")
    search_fields = ("program_normalized_name", "course_suffix")


@admin.register(ProgramOfferingBroadAggregate)
class ProgramOfferingBroadAggregateAdmin(admin.ModelAdmin):
    list_display = ("program_normalized_name", "offerings_count")
    search_fields = ("program_normalized_name",)


@admin.register(DedupCandidateGroup)
class DedupCandidateGroupAdmin(admin.ModelAdmin):
    list_display = ("institution", "institution_code", "normalized_name", "level", "campus", "rows_count")
    search_fields = ("institution_code", "institution_name", "normalized_name", "level", "campus")
    list_filter = ("level", "campus")


@admin.register(DedupSummary)
class DedupSummaryAdmin(admin.ModelAdmin):
    list_display = ("institution", "institution_code", "duplicate_groups", "duplicate_rows")
    search_fields = ("institution_code", "institution_name")


@admin.register(CodeCorrectionAudit)
class CodeCorrectionAuditAdmin(admin.ModelAdmin):
    list_display = ("program_code_before", "program_code_after", "correction_type", "institution_code")
    search_fields = ("program_code_before", "program_code_after", "correction_type", "institution_code", "group_key")
    list_filter = ("correction_type",)


@admin.register(ETLRun)
class ETLRunAdmin(admin.ModelAdmin):
    list_display = ("action", "created_at", "started_at", "finished_at")
    search_fields = ("action",)
    list_filter = ("action",)


@admin.register(DQReportEntry)
class DQReportEntryAdmin(admin.ModelAdmin):
    list_display = ("metric_name", "value", "scope", "run")
    search_fields = ("metric_name", "scope")


@admin.register(ClusterSubjectRule)
class ClusterSubjectRuleAdmin(admin.ModelAdmin):
    list_display = ("program_pattern",)
    search_fields = ("program_pattern",)


@admin.register(ProgramRequirementNormalized)
class ProgramRequirementNormalizedAdmin(admin.ModelAdmin):
    list_display = ("program",)
    search_fields = ("program__normalized_name",)


class ProgramRequirementOptionInline(admin.TabularInline):
    model = ProgramRequirementOption
    extra = 0
    fields = ("order", "subject", "subject_code", "min_grade")


@admin.register(ProgramRequirementGroup)
class ProgramRequirementGroupAdmin(admin.ModelAdmin):
    list_display = ("program", "order", "name", "pick", "options_preview")
    search_fields = ("program__normalized_name", "name")
    list_filter = ("pick",)
    inlines = [ProgramRequirementOptionInline]


@admin.register(ProgramRequirementOption)
class ProgramRequirementOptionAdmin(admin.ModelAdmin):
    list_display = ("group", "order", "subject", "subject_code", "min_grade")
    search_fields = ("group__program__normalized_name", "subject__name", "subject_code", "min_grade")


@admin.register(CourseSuffixMapping)
class CourseSuffixMappingAdmin(admin.ModelAdmin):
    list_display = ("course_suffix", "normalized_name", "field_name", "is_active", "updated_at")
    search_fields = ("course_suffix", "normalized_name", "field_name")
    list_filter = ("is_active",)
