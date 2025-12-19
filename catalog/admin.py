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
    actions = ("export_selected_as_csv", "import_from_mappings_csv",)

    def export_selected_as_csv(self, request, queryset):
        """Export selected CourseSuffixMapping rows as CSV."""
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = "attachment; filename=course_suffix_mappings.csv"
        writer = csv.writer(response)
        writer.writerow(["course_suffix", "normalized_name", "field_name", "is_active"]) 
        for obj in queryset:
            writer.writerow([obj.course_suffix, obj.normalized_name, obj.field_name, obj.is_active])
        return response
    export_selected_as_csv.short_description = "Export selected as CSV"

    def import_from_mappings_csv(self, request, queryset):
        """Import/Upsert mappings from the repo CSV at kuccps/mappings/course_suffix_map_overrides.csv.

        This is a convenience import (server-side path). For ad-hoc uploads, use the Django shell or a custom view.
        """
        base = Path(getattr(settings, "BASE_DIR", "."))
        rel = Path("scripts/etl/kuccps/mappings/course_suffix_map_overrides.csv")
        csv_path = (base / rel).resolve()
        if not csv_path.exists():
            self.message_user(request, f"CSV not found: {csv_path}", level=messages.ERROR)
            return
        from .models import CourseSuffixMapping
        created = 0
        updated = 0
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sfx = (row.get("course_suffix") or "").strip()
                if not sfx:
                    continue
                nm = (row.get("normalized_name") or "").strip()
                fd = (row.get("field_name") or "").strip()
                obj, was_created = CourseSuffixMapping.objects.get_or_create(course_suffix=sfx, defaults={
                    "normalized_name": nm or "",
                    "field_name": fd or "",
                    "is_active": True,
                })
                if was_created:
                    created += 1
                else:
                    changed = False
                    if nm and obj.normalized_name != nm:
                        obj.normalized_name = nm; changed = True
                    if fd and obj.field_name != fd:
                        obj.field_name = fd; changed = True
                    if not obj.is_active:
                        obj.is_active = True; changed = True
                    if changed:
                        obj.save(); updated += 1
        self.message_user(request, f"Import complete. Created={created}, Updated={updated}", level=messages.SUCCESS)
    import_from_mappings_csv.short_description = "Import from kuccps mappings CSV"
