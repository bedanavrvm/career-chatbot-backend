# Hand-written migration — 2026-02-27
# Changes:
#   1. UserProfile.email: add db_index=True
#   2. UserProfile.photo_url: increase max_length from 200 → 2048

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0002_onboardingprofile"),
    ]

    operations = [
        # 1. Index on email for admin lookups and Firebase claims matching.
        migrations.AlterField(
            model_name="userprofile",
            name="email",
            field=models.EmailField(
                blank=True,
                null=True,
                db_index=True,
                max_length=254,
            ),
        ),

        # 2. Increase photo_url max_length to accommodate long GCS/CDN URLs.
        #    Django's default URLField max_length is 200 which truncates Firebase
        #    storage URLs in practice.
        migrations.AlterField(
            model_name="userprofile",
            name="photo_url",
            field=models.URLField(
                blank=True,
                null=True,
                max_length=2048,
            ),
        ),
    ]
