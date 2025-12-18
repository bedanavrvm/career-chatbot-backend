from django.contrib import admin
from .models import Institution, Field, Subject, Program, YearlyCutoff, NormalizationRule, DedupMatch


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
    list_display = ("normalized_name", "institution", "level", "region")
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
