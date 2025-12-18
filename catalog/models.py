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
