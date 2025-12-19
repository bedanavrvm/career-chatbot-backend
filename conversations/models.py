import uuid
from typing import Optional
from django.db import models
from django.conf import settings
from django.utils import timezone
from cryptography.fernet import Fernet, InvalidToken
from datetime import timedelta
from django.db.models import Q


def _get_fernet() -> Optional[Fernet]:
    """Return a Fernet instance if PII_ENCRYPTION_KEY is configured; otherwise None."""
    key = getattr(settings, 'PII_ENCRYPTION_KEY', '') or ''
    key = key.strip()
    if not key:
        return None
    try:
        return Fernet(key.encode('utf-8'))
    except Exception:
        return None


def _encrypt_text(plaintext: str) -> str:
    """Encrypt text using Fernet if configured; otherwise return plaintext.
    The ciphertext returned is a URL-safe base64 string suitable for TextField storage.
    """
    if plaintext is None:
        return ''
    f = _get_fernet()
    if not f:
        return plaintext
    token = f.encrypt(plaintext.encode('utf-8'))
    return token.decode('utf-8')


def _decrypt_text(ciphertext: str) -> str:
    """Decrypt text using Fernet if configured; otherwise return the given string.
    If decryption fails, return an empty string to avoid leaking corrupted tokens.
    """
    if ciphertext is None:
        return ''
    f = _get_fernet()
    if not f:
        return ciphertext
    try:
        return f.decrypt(ciphertext.encode('utf-8')).decode('utf-8')
    except (InvalidToken, Exception):
        return ''


class Session(models.Model):
    """Represents a conversation session.

    - Minimal PII: optional external user identifier is stored encrypted.
    - TTL/expiry is tracked via expires_at and can be enforced by jobs or middleware.
    - FSM state and slot bag capture the deterministic flow context.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(max_length=20, default='active')  # active|closed|expired
    expires_at = models.DateTimeField(null=True, blank=True)
    fsm_state = models.CharField(max_length=64, default='greeting')
    slots = models.JSONField(default=dict)
    external_user_id_encrypted = models.TextField(blank=True, default='')
    idempotency_salt = models.CharField(max_length=64, blank=True, default='')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_activity_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['expires_at']),
        ]

    def set_external_user_id(self, value: Optional[str]) -> None:
        self.external_user_id_encrypted = _encrypt_text(value or '')

    def get_external_user_id(self) -> str:
        return _decrypt_text(self.external_user_id_encrypted)

    def ensure_ttl(self) -> None:
        """Ensure expires_at is set based on settings.CONV_SESSION_TTL_MINUTES if missing."""
        if not self.expires_at:
            ttl_min = int(getattr(settings, 'CONV_SESSION_TTL_MINUTES', 60) or 60)
            self.expires_at = timezone.now() + timedelta(minutes=ttl_min)


class Message(models.Model):
    """Stores messages in a session with encrypted content.

    - Idempotency: (session, idempotency_key) unique to prevent duplicate inserts.
    - NLP parse and FSM state snapshots are stored alongside.
    """
    ROLE_CHOICES = (
        ('user', 'user'),
        ('assistant', 'assistant'),
        ('system', 'system'),
    )

    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content_encrypted = models.TextField()
    idempotency_key = models.CharField(max_length=64, blank=True, null=True, default=None)
    nlp = models.JSONField(default=dict)
    fsm_state = models.CharField(max_length=64, blank=True, default='')
    confidence = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['session', 'created_at']),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['session', 'idempotency_key'],
                name='uniq_message_idem_per_session',
                condition=Q(idempotency_key__isnull=False),
            ),
        ]

    @property
    def content(self) -> str:
        return _decrypt_text(self.content_encrypted)

    @content.setter
    def content(self, value: str) -> None:
        self.content_encrypted = _encrypt_text(value or '')


class Profile(models.Model):
    """Aggregated profile derived from the conversation.

    - traits: RIASEC/interest indicators, personality facets
    - grades: subject->grade map used for eligibility checks
    - preferences: user preferences like region, cost, mode
    """
    session = models.OneToOneField(Session, on_delete=models.CASCADE, related_name='profile')
    traits = models.JSONField(default=dict)
    grades = models.JSONField(default=dict)
    preferences = models.JSONField(default=dict)
    version = models.CharField(max_length=16, default='v1')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
