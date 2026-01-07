import uuid

from django.test import TestCase, Client

from conversations.models import Message, Session


class ConversationsApiTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.session_id = uuid.uuid4()
        # In tests, verify_firebase_id_token returns deterministic claims (RUNNING_TESTS)
        self.auth_headers = {"HTTP_AUTHORIZATION": "Bearer test-token"}

    def test_post_message_creates_session(self):
        resp = self.client.post(
            f"/api/conversations/sessions/{self.session_id}/messages",
            data={"text": "Hello", "nlp_provider": "local"},
            content_type="application/json",
            **self.auth_headers,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("session", data)
        s = Session.objects.get(id=self.session_id)
        self.assertEqual(s.owner_uid, "test-user")

    def test_idempotency_key_returns_duplicate(self):
        idem = "k1"
        payload = {"text": "Hello", "idempotency_key": idem, "nlp_provider": "local"}

        r1 = self.client.post(
            f"/api/conversations/sessions/{self.session_id}/messages",
            data=payload,
            content_type="application/json",
            **self.auth_headers,
        )
        self.assertEqual(r1.status_code, 200)

        r2 = self.client.post(
            f"/api/conversations/sessions/{self.session_id}/messages",
            data=payload,
            content_type="application/json",
            **self.auth_headers,
        )
        self.assertEqual(r2.status_code, 200)
        d2 = r2.json()
        self.assertTrue(d2.get("duplicate"))

        s = Session.objects.get(id=self.session_id)
        self.assertEqual(Message.objects.filter(session=s, idempotency_key=idem).count(), 1)
