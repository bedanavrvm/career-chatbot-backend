from django.db import models


class OnetFieldOccupationMapping(models.Model):
    field = models.ForeignKey('catalog.Field', on_delete=models.CASCADE, related_name='onet_mappings')
    occupation_code = models.CharField(max_length=10)
    weight = models.DecimalField(max_digits=8, decimal_places=3, null=True, blank=True)
    notes = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['field_id', '-weight', 'occupation_code']
        unique_together = (('field', 'occupation_code'),)
        indexes = [models.Index(fields=['occupation_code'], name='onet_field_occ_code_idx')]

    def __str__(self) -> str:
        try:
            return f"{self.field.slug} -> {self.occupation_code}"
        except Exception:
            return f"{self.field_id} -> {self.occupation_code}"
