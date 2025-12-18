from django.db import models
from django.utils.text import slugify


class TimestampedModel(models.Model):
    """Abstract base with created/updated timestamps for all domain tables."""
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Institution(TimestampedModel):
    """Higher learning institution (University/College) including region metadata."""
    code = models.CharField(max_length=32, unique=True)
    name = models.CharField(max_length=255)
    alias = models.CharField(max_length=255, blank=True)
    region = models.CharField(max_length=64, blank=True)
    county = models.CharField(max_length=64, blank=True)
    website = models.URLField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["region"]),
            models.Index(fields=["county"]),
        ]
        ordering = ["name"]

    def __str__(self) -> str:
        return f"{self.name} ({self.code})"


class Field(TimestampedModel):
    """Broad field of study (e.g., Engineering, Medicine). Supports simple hierarchy."""
    name = models.CharField(max_length=128, unique=True)
    slug = models.SlugField(max_length=140, unique=True, blank=True)
    parent = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL, related_name="children")
    description = models.TextField(blank=True)

    class Meta:
        ordering = ["name"]

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        return super().save(*args, **kwargs)

    def __str__(self) -> str:
        return self.name


class Subject(TimestampedModel):
    """KCSE subject taxonomy (canonical codes/names)."""
    code = models.CharField(max_length=32, unique=True)
    name = models.CharField(max_length=128)
    group = models.CharField(max_length=64, blank=True)
    alt_codes = models.JSONField(default=list, blank=True)

    class Meta:
        indexes = [models.Index(fields=["group"])]
        ordering = ["code"]

    def __str__(self) -> str:
        return f"{self.code}: {self.name}"


class ProgramLevel(models.TextChoices):
    BACHELOR = "bachelor", "Bachelor"
    DIPLOMA = "diploma", "Diploma"
    CERTIFICATE = "certificate", "Certificate"


class Program(TimestampedModel):
    """Academic program offered by an institution with normalized metadata and requirements."""
    institution = models.ForeignKey(Institution, on_delete=models.CASCADE, related_name="programs")
    field = models.ForeignKey(Field, on_delete=models.SET_NULL, null=True, related_name="programs")
    code = models.CharField(max_length=64, blank=True)
    name = models.CharField(max_length=255)
    normalized_name = models.CharField(max_length=255, db_index=True)
    level = models.CharField(max_length=32, choices=ProgramLevel.choices)
    campus = models.CharField(max_length=128, blank=True)
    region = models.CharField(max_length=64, blank=True)
    duration_years = models.DecimalField(max_digits=4, decimal_places=1, null=True, blank=True)
    award = models.CharField(max_length=128, blank=True)
    mode = models.CharField(max_length=64, blank=True)
    subject_requirements = models.JSONField(default=dict, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = (
            ("institution", "normalized_name", "level", "campus"),
        )
        indexes = [
            models.Index(fields=["level"]),
            models.Index(fields=["region"]),
            models.Index(fields=["institution", "level"]),
        ]
        ordering = ["normalized_name"]

    def __str__(self) -> str:
        return f"{self.normalized_name} @ {self.institution.code}"


class YearlyCutoff(TimestampedModel):
    """Yearly cutoff score per program; optional capacity and notes."""
    program = models.ForeignKey(Program, on_delete=models.CASCADE, related_name="cutoffs")
    year = models.PositiveIntegerField()
    cutoff = models.DecimalField(max_digits=6, decimal_places=3)
    capacity = models.PositiveIntegerField(null=True, blank=True)
    notes = models.CharField(max_length=255, blank=True)

    class Meta:
        unique_together = (("program", "year"),)
        indexes = [models.Index(fields=["year"])]
        ordering = ["-year"]

    def __str__(self) -> str:
        return f"{self.program_id}:{self.year}={self.cutoff}"


class NormalizationRule(TimestampedModel):
    """Maps messy source values to normalized values for consistent taxonomy."""
    TYPE_CHOICES = (
        ("PROGRAM_NAME", "Program Name"),
        ("INSTITUTION_NAME", "Institution Name"),
        ("LEVEL", "Level"),
        ("SUBJECT", "Subject"),
        ("REGION", "Region"),
    )
    type = models.CharField(max_length=32, choices=TYPE_CHOICES)
    source_value = models.CharField(max_length=255, db_index=True)
    normalized_value = models.CharField(max_length=255)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = (("type", "source_value"),)
        ordering = ["type", "source_value"]

    def __str__(self) -> str:
        return f"{self.type}:{self.source_value}→{self.normalized_value}"


class DedupMatch(TimestampedModel):
    """Records duplicate program resolution (merge mapping)."""
    master_program = models.ForeignKey(Program, on_delete=models.CASCADE, related_name="dedup_master")
    duplicate_program = models.ForeignKey(Program, on_delete=models.CASCADE, related_name="dedup_duplicate", unique=True)
    reason = models.CharField(max_length=255, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.duplicate_program_id} -> {self.master_program_id}"


# -----------------------------------------
# New models (derived from ETL CSV artifacts)
# -----------------------------------------


class InstitutionCampus(TimestampedModel):
    """Campus-level geo for an Institution (from mappings/institutions_campuses.csv)."""
    institution = models.ForeignKey(Institution, on_delete=models.CASCADE, related_name="campuses")
    campus = models.CharField(max_length=128)
    town = models.CharField(max_length=128, blank=True)
    county = models.CharField(max_length=64, blank=True)
    region = models.CharField(max_length=64, blank=True)

    class Meta:
        unique_together = (("institution", "campus"),)
        indexes = [models.Index(fields=["region"]), models.Index(fields=["county"])]
        ordering = ["institution", "campus"]

    def __str__(self) -> str:
        return f"{self.institution.code}:{self.campus}"


class ProgramOfferingAggregate(TimestampedModel):
    """Aggregate of unique institutions offering a course variant (program_offerings.csv)."""
    program_normalized_name = models.CharField(max_length=255, db_index=True)
    course_suffix = models.CharField(max_length=16, blank=True)
    offerings_count = models.PositiveIntegerField()

    class Meta:
        unique_together = (("program_normalized_name", "course_suffix"),)
        ordering = ["program_normalized_name", "course_suffix"]

    def __str__(self) -> str:
        return f"{self.program_normalized_name} [{self.course_suffix}]: {self.offerings_count}"


class ProgramOfferingBroadAggregate(TimestampedModel):
    """Aggregate across degree families (program_offerings_broad.csv)."""
    program_normalized_name = models.CharField(max_length=255, unique=True)
    offerings_count = models.PositiveIntegerField()

    class Meta:
        ordering = ["program_normalized_name"]

    def __str__(self) -> str:
        return f"{self.program_normalized_name}: {self.offerings_count}"


class DedupCandidateGroup(TimestampedModel):
    """Duplicate group suggested by ETL (from dedup_candidates.csv)."""
    institution = models.ForeignKey(Institution, null=True, blank=True, on_delete=models.SET_NULL, related_name="dedup_candidate_groups")
    institution_code = models.CharField(max_length=32, blank=True)
    institution_name = models.CharField(max_length=255, blank=True)
    normalized_name = models.CharField(max_length=255)
    level = models.CharField(max_length=32, choices=ProgramLevel.choices)
    campus = models.CharField(max_length=128, blank=True)
    rows_count = models.PositiveIntegerField(default=0)
    program_codes = models.JSONField(default=list, blank=True)
    name_variants = models.JSONField(default=list, blank=True)
    suggested_master_program_code = models.CharField(max_length=64, blank=True)

    class Meta:
        unique_together = (("institution", "normalized_name", "level", "campus"),)
        indexes = [models.Index(fields=["level"]), models.Index(fields=["campus"])]
        ordering = ["-rows_count"]

    def __str__(self) -> str:
        return f"dup:{self.institution_code}:{self.normalized_name}:{self.level}:{self.campus}"


class DedupSummary(TimestampedModel):
    """Per-institution summary of duplicate groups/rows (from dedup_summary.csv)."""
    institution = models.ForeignKey(Institution, null=True, blank=True, on_delete=models.SET_NULL, related_name="dedup_summaries")
    institution_code = models.CharField(max_length=32, blank=True)
    institution_name = models.CharField(max_length=255, blank=True)
    duplicate_groups = models.PositiveIntegerField(default=0)
    duplicate_rows = models.PositiveIntegerField(default=0)

    class Meta:
        unique_together = (("institution",),)
        ordering = ["-duplicate_rows"]

    def __str__(self) -> str:
        return f"summary:{self.institution_code} g={self.duplicate_groups} r={self.duplicate_rows}"


class CodeCorrectionAudit(TimestampedModel):
    """Audit for program code normalization (_code_corrections.csv)."""
    program_code_before = models.CharField(max_length=64, blank=True)
    program_code_after = models.CharField(max_length=64, blank=True)
    correction_type = models.CharField(max_length=64, blank=True)
    institution_code = models.CharField(max_length=32, blank=True)
    group_key = models.CharField(max_length=128, blank=True)
    reason = models.CharField(max_length=255, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [models.Index(fields=["correction_type"])]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.program_code_before}->{self.program_code_after} ({self.correction_type})"


class ETLRun(TimestampedModel):
    """Basic audit trail for ETL actions and their computed stats."""
    action = models.CharField(max_length=64)
    config_path = models.CharField(max_length=255, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    stats = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [models.Index(fields=["action", "created_at"])]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.action}@{self.created_at:%Y-%m-%d %H:%M:%S}"


class DQReportEntry(TimestampedModel):
    """Data quality metric line item (from dq_report.csv)."""
    run = models.ForeignKey(ETLRun, null=True, blank=True, on_delete=models.SET_NULL, related_name="dq_entries")
    metric_name = models.CharField(max_length=128)
    value = models.CharField(max_length=64)
    scope = models.CharField(max_length=64, blank=True)
    extra = models.JSONField(default=dict, blank=True)

    class Meta:
        indexes = [models.Index(fields=["metric_name"])]
        ordering = ["metric_name"]

    def __str__(self) -> str:
        return f"{self.metric_name}={self.value}"


class ClusterSubjectRule(TimestampedModel):
    """Eligibility cluster rule from mappings/cluster_subjects.csv."""
    program_pattern = models.CharField(max_length=255, db_index=True)
    subjects_grammar = models.TextField()

    class Meta:
        ordering = ["program_pattern"]

    def __str__(self) -> str:
        return self.program_pattern


class ProgramRequirementNormalized(TimestampedModel):
    """Optional normalized store for subject requirements extracted from Program JSON.
    This is a lightweight denormalization for future advanced eligibility.
    """
    program = models.ForeignKey(Program, on_delete=models.CASCADE, related_name="normalized_requirements")
    required = models.JSONField(default=list, blank=True)
    groups = models.JSONField(default=list, blank=True)
    notes = models.CharField(max_length=255, blank=True)

    class Meta:
        ordering = ["program"]

    def __str__(self) -> str:
        return f"reqs:{self.program_id}"
