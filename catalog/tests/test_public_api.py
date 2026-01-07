import os
from pathlib import Path
from tempfile import TemporaryDirectory
from django.test import TestCase, Client
from django.contrib.auth import get_user_model


class PublicApiTests(TestCase):
    def setUp(self):
        self.client = Client()
        # Ensure CSRF is bypassed for admin ETL endpoints during tests
        os.environ["DISABLE_CSRF_DEV"] = "true"

        from catalog.models import Institution, Field, Program

        inst = Institution.objects.create(
            code="1170",
            name="MACHAKOS UNIVERSITY",
            alias="MKSU",
            region="Eastern",
            county="Machakos",
            website="https://mksu.ac.ke",
        )
        field = Field.objects.create(name="Engineering")
        Program.objects.create(
            institution=inst,
            field=field,
            code="1170114",
            name="BSc Civil Engineering",
            normalized_name="BSC CIVIL ENGINEERING",
            level="bachelor",
            region="Eastern",
            campus="Main",
            award="BSc",
            subject_requirements={},
            metadata={},
        )

    def _setup_admin_etl_dirs(self):
        tmp = TemporaryDirectory()
        base = Path(tmp.name)
        processed = base / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        # Point admin ETL pages at this processed dir so uploads/logs do not write into the repo.
        os.environ["KUCCPS_PROCESSED_DIR"] = str(processed)
        return tmp

    def test_institutions_endpoint(self):
        resp = self.client.get("/api/etl/institutions", {"q": "machakos"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data.get("count", 0), 1)

    def test_fields_endpoint(self):
        resp = self.client.get("/api/etl/fields", {"q": "engineer"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data.get("count", 0), 1)

    def test_programs_endpoint(self):
        resp = self.client.get("/api/etl/programs", {"q": "civil", "level": "bachelor"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data.get("count", 0), 1)
        self.assertTrue(len(data.get("results", [])) <= 20)

    def test_search_endpoint(self):
        resp = self.client.get("/api/etl/search", {"q": "engineering"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("programs", data)
        self.assertIn("institutions", data)
        self.assertIn("fields", data)

    def test_admin_upload_and_process_pages(self):
        tmp = self._setup_admin_etl_dirs()
        try:
            # Create staff user and login
            User = get_user_model()
            user = User.objects.create_user(username="staff", password="pass", is_staff=True)
            self.client.login(username="staff", password="pass")

            # GET upload page
            r = self.client.get("/admin/etl/upload")
            self.assertEqual(r.status_code, 200)

            # POST upload a small file
            from io import BytesIO
            file_content = BytesIO(b"%PDF-1.4 test")
            file_content.name = "test.pdf"
            r2 = self.client.post("/admin/etl/upload", {"file": file_content})
            self.assertEqual(r2.status_code, 200)

            # GET process page
            r3 = self.client.get("/admin/etl/process")
            self.assertEqual(r3.status_code, 200)

            # POST process
            r4 = self.client.post("/admin/etl/process", {"action": "transform-normalize"})
            self.assertEqual(r4.status_code, 200)
        finally:
            tmp.cleanup()
