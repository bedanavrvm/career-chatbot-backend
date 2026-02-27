# Hand-written migration — 2026-02-27
# Changes:
#   1. Session.last_activity_at: remove auto_now=True → plain DateTimeField(null=True)
#   2. New index on (owner_uid, last_activity_at) for the session-list ORDER BY
#   3. New index on (session, idempotency_key) for idempotency dedup lookups in Message

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("conversations", "0002_session_owner_uid"),
    ]

    operations = [
        # 1. Change last_activity_at from auto_now to a plain nullable field.
        #    auto_now silently ignored update_fields=['last_activity_at'] calls
        #    throughout views.py, meaning the field was never actually updated.
        migrations.AlterField(
            model_name="session",
            name="last_activity_at",
            field=models.DateTimeField(blank=True, null=True, db_index=True),
        ),

        # 2. Composite index on (owner_uid, last_activity_at) — used in
        #    Session.objects.filter(owner_uid=uid).order_by('-last_activity_at')
        migrations.AddIndex(
            model_name="session",
            index=models.Index(
                fields=["owner_uid", "last_activity_at"],
                name="conv_sess_owner_activity_idx",
            ),
        ),

        # 3. Composite index on (session_id, idempotency_key) — used in every
        #    post_message call to detect duplicate messages.
        migrations.AddIndex(
            model_name="message",
            index=models.Index(
                fields=["session", "idempotency_key"],
                name="conv_msg_idem_idx",
            ),
        ),
    ]
