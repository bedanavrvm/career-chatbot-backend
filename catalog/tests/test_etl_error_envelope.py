from django.test import TestCase, Client


class EtlErrorEnvelopeTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_search_requires_q_envelope(self):
        resp = self.client.get('/api/etl/search')
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertIn('detail', data)
        self.assertIn('code', data)
        self.assertIn('fields', data)
        self.assertEqual(data.get('code'), 'validation_error')
        self.assertIn('q', (data.get('fields') or {}))

    def test_eligibility_requires_program_code_and_grades_envelope(self):
        resp = self.client.post('/api/etl/eligibility', data='{}', content_type='application/json')
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertIn('detail', data)
        self.assertIn('code', data)
        self.assertEqual(data.get('code'), 'validation_error')
        fields = data.get('fields') or {}
        self.assertIn('program_code', fields)
        self.assertIn('grades', fields)
