from django.test import TestCase, Client


class CatalogStatusApiTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_catalog_status_endpoint(self):
        resp = self.client.get("/api/catalog/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("catalog", data)
        self.assertIn("etl", data)
        self.assertIn("rag", data)
