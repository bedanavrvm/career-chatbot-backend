from django.db import models


class OnetOccupation(models.Model):
    onetsoc_code = models.CharField(max_length=10, primary_key=True)
    title = models.CharField(max_length=150)
    description = models.CharField(max_length=1000)

    class Meta:
        db_table = 'occupation_data'
        managed = False


class OnetContentElement(models.Model):
    element_id = models.CharField(max_length=20, primary_key=True)
    element_name = models.CharField(max_length=150)
    description = models.CharField(max_length=1500)

    class Meta:
        db_table = 'content_model_reference'
        managed = False


class OnetScale(models.Model):
    scale_id = models.CharField(max_length=3, primary_key=True)
    scale_name = models.CharField(max_length=50)
    minimum = models.DecimalField(max_digits=4, decimal_places=0)
    maximum = models.DecimalField(max_digits=6, decimal_places=0)

    class Meta:
        db_table = 'scales_reference'
        managed = False


class OnetInterest(models.Model):
    id = models.BigAutoField(primary_key=True)
    onetsoc_code = models.ForeignKey(OnetOccupation, on_delete=models.CASCADE, db_column='onetsoc_code')
    element_id = models.ForeignKey(OnetContentElement, on_delete=models.PROTECT, db_column='element_id')
    scale_id = models.ForeignKey(OnetScale, on_delete=models.PROTECT, db_column='scale_id')
    data_value = models.DecimalField(max_digits=5, decimal_places=2)
    date_updated = models.DateField()
    domain_source = models.CharField(max_length=30)

    class Meta:
        db_table = 'interests'
        managed = False
        unique_together = (('onetsoc_code', 'element_id', 'scale_id', 'date_updated', 'domain_source'),)


class OnetSkill(models.Model):
    id = models.BigAutoField(primary_key=True)
    onetsoc_code = models.ForeignKey(OnetOccupation, on_delete=models.CASCADE, db_column='onetsoc_code')
    element_id = models.ForeignKey(OnetContentElement, on_delete=models.PROTECT, db_column='element_id')
    scale_id = models.ForeignKey(OnetScale, on_delete=models.PROTECT, db_column='scale_id')
    data_value = models.DecimalField(max_digits=5, decimal_places=2)
    n = models.DecimalField(max_digits=4, decimal_places=0, null=True, blank=True)
    standard_error = models.DecimalField(max_digits=7, decimal_places=4, null=True, blank=True)
    lower_ci_bound = models.DecimalField(max_digits=7, decimal_places=4, null=True, blank=True)
    upper_ci_bound = models.DecimalField(max_digits=7, decimal_places=4, null=True, blank=True)
    recommend_suppress = models.CharField(max_length=1, null=True, blank=True)
    not_relevant = models.CharField(max_length=1, null=True, blank=True)
    date_updated = models.DateField()
    domain_source = models.CharField(max_length=30)

    class Meta:
        db_table = 'skills'
        managed = False
        unique_together = (('onetsoc_code', 'element_id', 'scale_id', 'date_updated', 'domain_source'),)


class OnetTaskStatement(models.Model):
    task_id = models.BigIntegerField(primary_key=True)
    onetsoc_code = models.ForeignKey(OnetOccupation, on_delete=models.CASCADE, db_column='onetsoc_code')
    task = models.CharField(max_length=1000)
    task_type = models.CharField(max_length=12, null=True, blank=True)
    incumbents_responding = models.DecimalField(max_digits=4, decimal_places=0, null=True, blank=True)
    date_updated = models.DateField()
    domain_source = models.CharField(max_length=30)

    class Meta:
        db_table = 'task_statements'
        managed = False


class OnetRelatedOccupation(models.Model):
    id = models.BigAutoField(primary_key=True)
    onetsoc_code = models.ForeignKey(OnetOccupation, on_delete=models.CASCADE, db_column='onetsoc_code', related_name='related_from')
    related_onetsoc_code = models.ForeignKey(OnetOccupation, on_delete=models.CASCADE, db_column='related_onetsoc_code', related_name='related_to')
    relatedness_tier = models.CharField(max_length=50)
    related_index = models.DecimalField(max_digits=3, decimal_places=0)

    class Meta:
        db_table = 'related_occupations'
        managed = False
        unique_together = (('onetsoc_code', 'related_onetsoc_code', 'relatedness_tier', 'related_index'),)
