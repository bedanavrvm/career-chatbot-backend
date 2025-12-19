import json
from typing import Any, Dict
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.conf import settings
from .models import Session, Message, Profile
from .fsm import next_turn


def _now():
    return timezone.now()


def _ensure_session_ttl(sess: Session) -> None:
    sess.ensure_ttl()
    sess.save(update_fields=['expires_at'])


def _serialize_message(m: Message) -> Dict[str, Any]:
    return {
        'id': m.id,
        'role': m.role,
        'content': m.content,
        'fsm_state': m.fsm_state,
        'confidence': m.confidence,
        'created_at': m.created_at.isoformat(),
        'nlp': m.nlp or {},
    }


def _serialize_session(s: Session, limit_messages: int = 20) -> Dict[str, Any]:
    msgs = list(s.messages.order_by('-created_at')[:limit_messages])
    msgs = list(reversed(msgs))
    return {
        'id': str(s.id),
        'status': s.status,
        'fsm_state': s.fsm_state,
        'slots': s.slots,
        'expires_at': s.expires_at.isoformat() if s.expires_at else None,
        'messages': [_serialize_message(m) for m in msgs],
    }


@csrf_exempt
def get_session(request, session_id):
    """GET /api/conversations/sessions/{id}
    Returns session summary and recent messages. Creates a new session on first access.
    """
    if request.method != 'GET':
        return HttpResponseBadRequest('GET required')
    try:
        try:
            s = Session.objects.get(id=session_id)
        except Session.DoesNotExist:
            s = Session(id=session_id)
            s.ensure_ttl()
            s.save()
        _ensure_session_ttl(s)
        return JsonResponse(_serialize_session(s))
    except Exception as e:
        return JsonResponse({'detail': str(e)}, status=500)


@csrf_exempt
def delete_session(request, session_id):
    """DELETE/POST /api/conversations/sessions/{id}/delete
    Deletes a conversation session and its messages. Returns 204 on success.
    """
    if request.method not in ('DELETE', 'POST'):
        return HttpResponseBadRequest('DELETE or POST required')
    try:
        try:
            s = Session.objects.get(id=session_id)
        except Session.DoesNotExist:
            return JsonResponse({}, status=204)
        s.delete()
        return JsonResponse({}, status=204)
    except Exception as e:
        return JsonResponse({'detail': str(e)}, status=500)


@csrf_exempt
def post_message(request, session_id):
    """POST /api/conversations/sessions/{id}/messages
    Body: { text: str, idempotency_key?: str, user_id?: str }
    Creates the session if it does not exist yet (with TTL), enforces idempotency, and appends the user message.
    Returns a stub assistant reply and current FSM state.
    """
    if request.method != 'POST':
        return HttpResponseBadRequest('POST required')
    try:
        data = json.loads(request.body.decode('utf-8') or '{}')
        text = (data.get('text') or '').strip()
        idem = (data.get('idempotency_key') or '').strip()
        user_id = (data.get('user_id') or '').strip()
        if not text:
            return JsonResponse({'detail': 'text is required'}, status=400)
        try:
            s = Session.objects.get(id=session_id)
        except Session.DoesNotExist:
            s = Session(id=session_id)
        if user_id:
            s.set_external_user_id(user_id)
        s.ensure_ttl()
        s.last_activity_at = _now()
        s.save()

        # Idempotency: return prior message result if same key exists (user role only)
        if idem:
            try:
                prior = Message.objects.get(session=s, idempotency_key=idem)
                # Return the current session snapshot
                return JsonResponse({
                    'session': _serialize_session(s),
                    'duplicate': True,
                })
            except Message.DoesNotExist:
                pass

        # Store user message (encrypted)
        um = Message(session=s, role='user', idempotency_key=idem)
        um.content = text
        um.fsm_state = s.fsm_state
        # Compute next turn using FSM+NLP
        tr = next_turn(s, text)
        # Persist NLP payload on the user message for observability
        um.nlp = tr.nlp_payload or {}
        um.save()

        # Update session state and slots
        s.fsm_state = tr.next_state
        s.slots = tr.slots
        s.last_activity_at = _now()
        s.save(update_fields=['fsm_state', 'slots', 'last_activity_at'])

        # Assistant reply
        am = Message(session=s, role='assistant', fsm_state=tr.next_state)
        am.content = tr.reply
        am.confidence = tr.confidence
        am.nlp = tr.nlp_payload or {}
        am.save()

        s.last_activity_at = _now()
        s.save(update_fields=['last_activity_at'])

        return JsonResponse({'session': _serialize_session(s)})
    except Exception as e:
        return JsonResponse({'detail': str(e)}, status=500)


@csrf_exempt
def post_profile(request):
    """POST /api/conversations/profile
    Body: { session_id: UUID, traits?: {}, grades?: {}, preferences?: {}, version?: str }
    Upserts the profile for the session.
    """
    if request.method != 'POST':
        return HttpResponseBadRequest('POST required')
    try:
        data = json.loads(request.body.decode('utf-8') or '{}')
        session_id = data.get('session_id')
        if not session_id:
            return JsonResponse({'detail': 'session_id is required'}, status=400)
        try:
            s = Session.objects.get(id=session_id)
        except Session.DoesNotExist:
            s = Session(id=session_id)
            s.ensure_ttl()
            s.save()

        prof, _ = Profile.objects.get_or_create(session=s)
        if 'traits' in data and isinstance(data['traits'], dict):
            prof.traits = data['traits']
        if 'grades' in data and isinstance(data['grades'], dict):
            prof.grades = data['grades']
        if 'preferences' in data and isinstance(data['preferences'], dict):
            prof.preferences = data['preferences']
        if 'version' in data and isinstance(data['version'], str):
            prof.version = data['version']
        prof.save()
        return JsonResponse({
            'session_id': str(s.id),
            'profile': {
                'traits': prof.traits,
                'grades': prof.grades,
                'preferences': prof.preferences,
                'version': prof.version,
            }
        })
    except Exception as e:
        return JsonResponse({'detail': str(e)}, status=500)
