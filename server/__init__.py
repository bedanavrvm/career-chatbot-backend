# This makes the Celery application available as server.celery.app
from .celery import app as celery_app  # noqa: F401

__all__ = ('celery_app',)
