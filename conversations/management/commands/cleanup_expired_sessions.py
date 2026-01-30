from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from conversations.models import Session


class Command(BaseCommand):
    help = 'Delete expired conversation sessions (and cascaded messages/profiles).'

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true', help='Only report how many sessions would be deleted')
        parser.add_argument('--limit', type=int, default=0, help='Maximum number of sessions to delete (0 = no limit)')

    def handle(self, *args, **options):
        dry_run = bool(options.get('dry_run'))
        try:
            limit = int(options.get('limit') or 0)
        except Exception:
            limit = 0

        now = timezone.now()
        qs = Session.objects.filter(expires_at__isnull=False, expires_at__lt=now).order_by('expires_at')
        total = int(qs.count())
        if limit > 0:
            qs = qs[: max(0, limit)]

        to_delete = list(qs)
        self.stdout.write(f'Expired sessions found: {total}')
        self.stdout.write(f'Deleting: {len(to_delete)} (dry_run={dry_run})')

        if dry_run or not to_delete:
            return

        ids = [s.id for s in to_delete]
        with transaction.atomic():
            deleted = Session.objects.filter(id__in=ids).delete()

        self.stdout.write(f'Delete result: {deleted}')
