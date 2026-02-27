"""
conversations/tasks.py
Celery tasks for async LLM message processing.

When CELERY_BROKER_URL is set, the post_message endpoint offloads the
LLM/NLP call to a worker process and returns a task_id immediately.
The frontend polls GET /api/conversations/tasks/{task_id}/status until done.

When CELERY_BROKER_URL is NOT set (development), CELERY_TASK_ALWAYS_EAGER=True
means the task runs synchronously within the same process — nothing changes
from the caller's perspective.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from celery import shared_task
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    name='conversations.process_message',
    max_retries=2,
    default_retry_delay=2,
    acks_late=True,  # Re-queue on worker crash
)
def process_message_task(
    self,
    *,
    session_id: str,
    uid: str,
    text: str,
    idempotency_key: str = '',
    nlp_provider: str = '',
    use_planner: bool = False,
    shadow_mode: bool = False,
) -> Dict[str, Any]:
    """
    Process a single conversation message asynchronously.

    Returns a dict with keys:
        session   – serialized session snapshot
        turn_recommendations – optional recommendation payload
        error     – present only on failure
    """
    from django.db import transaction, IntegrityError
    from .models import Session, Message, Profile
    from .fsm import next_turn
    from .orchestrator import planner_turn
    # _serialize_session lives in views.py (not serializers.py)
    from .views import _serialize_session as _ser_sess

    now = timezone.now

    try:
        # --- Atomic get-or-create (same logic as the sync view) ---
        with transaction.atomic():
            try:
                s = Session.objects.select_for_update().get(id=session_id)
            except Session.DoesNotExist:
                try:
                    s = Session.objects.create(id=session_id, owner_uid=uid)
                    s.set_external_user_id(uid)
                    s.ensure_ttl()
                    s.last_activity_at = now()
                    s.save()
                except IntegrityError:
                    s = Session.objects.select_for_update().get(id=session_id)

        if s.owner_uid and s.owner_uid != uid:
            return {'error': 'forbidden', 'session': None}

        if not s.owner_uid:
            s.owner_uid = uid
            s.set_external_user_id(uid)
        s.ensure_ttl()
        s.last_activity_at = now()
        s.save()

        # --- Idempotency check ---
        if idempotency_key:
            try:
                Message.objects.get(session=s, idempotency_key=idempotency_key)
                # Already processed — return current state
                return {'session': _ser_sess(s), 'duplicate': True, 'turn_recommendations': None}
            except Message.DoesNotExist:
                pass

        # --- Store user message ---
        um = Message(session=s, role='user', idempotency_key=idempotency_key)
        um.content = text
        um.fsm_state = s.fsm_state

        # --- Run FSM / LLM ---
        tr = None
        planner_tr = None
        if use_planner:
            try:
                planner_tr = planner_turn(s, text, uid=uid, provider_override=nlp_provider)
            except Exception:
                logger.exception('process_message_task: planner_turn failed for session=%s', session_id)
            if not shadow_mode:
                tr = planner_tr

        if tr is None:
            tr = next_turn(s, text, provider_override=nlp_provider)

        if planner_tr is not None and shadow_mode:
            nlp_shadow = planner_tr.nlp_payload or {}
            nlp_main = tr.nlp_payload or {}
            tr.nlp_payload = {**nlp_main, 'shadow_planner': nlp_shadow}

        um.nlp = tr.nlp_payload or {}
        um.save()

        # --- Update session ---
        s.fsm_state = tr.next_state
        s.slots = tr.slots
        s.last_activity_at = now()
        s.save(update_fields=['fsm_state', 'slots', 'last_activity_at'])

        # --- Store assistant reply ---
        am = Message(session=s, role='assistant', fsm_state=tr.next_state)
        am.content = tr.reply
        am.confidence = tr.confidence
        am.nlp = tr.nlp_payload or {}
        am.save()

        s.last_activity_at = now()
        s.save(update_fields=['last_activity_at'])

        turn_recs = None
        try:
            nlp_payload = tr.nlp_payload or {}
            if isinstance(nlp_payload, dict) and isinstance(nlp_payload.get('turn_recommendations'), dict):
                turn_recs = nlp_payload.get('turn_recommendations')
        except Exception:
            pass

        return {'session': _ser_sess(s), 'turn_recommendations': turn_recs}

    except Exception as exc:
        logger.exception('process_message_task failed for session=%s', session_id)
        # Retry on transient errors (network, DB blip)
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {'error': str(exc), 'session': None, 'turn_recommendations': None}


@shared_task(
    name='conversations.cleanup_expired_sessions',
    ignore_result=True,
)
def cleanup_expired_sessions() -> None:
    """
    Periodic maintenance task: delete sessions (and cascade: messages, profiles)
    whose TTL has elapsed.  Runs on Celery Beat; configure via CELERY_BEAT_SCHEDULE
    in settings or leave the default hourly schedule below.
    """
    from .models import Session

    cutoff = timezone.now()
    qs = Session.objects.filter(expires_at__lt=cutoff)
    count, _ = qs.delete()
    if count:
        logger.info('cleanup_expired_sessions: deleted %d expired session(s)', count)
    else:
        logger.debug('cleanup_expired_sessions: no expired sessions to clean up')


# ---------------------------------------------------------------------------
# Celery Beat schedule (registered programmatically so it works without
# a separate django-celery-beat database table).
# Override any entry by setting CELERY_BEAT_SCHEDULE in Django settings.
# ---------------------------------------------------------------------------
try:
    from server.celery import app as _celery_app  # type: ignore[import]

    _celery_app.conf.beat_schedule.setdefault(
        'conversations-cleanup-expired-sessions',
        {
            'task': 'conversations.cleanup_expired_sessions',
            'schedule': 3600,  # every hour
        },
    )
except Exception:
    pass  # Beat schedule is optional; tasks still work without it
