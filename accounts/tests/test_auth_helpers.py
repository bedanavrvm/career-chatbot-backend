from unittest.mock import patch

from django.test import TestCase, RequestFactory
from rest_framework.response import Response

from accounts.auth import require_firebase_uid


class RequireFirebaseUidTests(TestCase):
    def setUp(self):
        self.rf = RequestFactory()

    def test_missing_bearer_token_returns_401(self):
        req = self.rf.get("/api/secure-ping/")
        uid, resp = require_firebase_uid(
            req,
            response_factory=lambda data, status: Response(data, status=status),
        )
        self.assertIsNone(uid)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.status_code, 401)

    def test_valid_token_returns_uid(self):
        req = self.rf.get("/api/secure-ping/", HTTP_AUTHORIZATION="Bearer testtoken")
        with patch("accounts.auth.verify_firebase_id_token") as mock_verify:
            mock_verify.return_value = ({"uid": "u123"}, None, None)
            uid, resp = require_firebase_uid(
                req,
                response_factory=lambda data, status: Response(data, status=status),
            )
        self.assertEqual(uid, "u123")
        self.assertIsNone(resp)

    def test_invalid_token_returns_401(self):
        req = self.rf.get("/api/secure-ping/", HTTP_AUTHORIZATION="Bearer bad")
        with patch("accounts.auth.verify_firebase_id_token") as mock_verify:
            mock_verify.return_value = (None, "Invalid token", 401)
            uid, resp = require_firebase_uid(
                req,
                response_factory=lambda data, status: Response(data, status=status),
            )
        self.assertIsNone(uid)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.status_code, 401)
        self.assertEqual((resp.data or {}).get("detail"), "Invalid token")
