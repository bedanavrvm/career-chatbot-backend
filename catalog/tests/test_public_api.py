import os
import json
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from django.test import TestCase, Client, override_settings
from django.contrib.auth import get_user_model


class PublicApiTests(TestCase):
    def setUp(self):
        self.client = Client()
        # Ensure CSRF is bypassed for admin ETL endpoints during tests
        os.environ["DISABLE_CSRF_DEV"] = "true"

    def _setup_processed_dir(self):
        tmp = TemporaryDirectory()
        base = Path(tmp.name)
        processed = base / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        # Institutions
        with (processed / "institutions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["code", "name", "alias", "region", "county", "country", "website"])
            w.writerow(["1170", "MACHAKOS UNIVERSITY", "MKSU", "Eastern", "Machakos", "Kenya", "https://mksu.ac.ke"]) 
        # Fields
        with (processed / "fields.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "parent", "description"])
            w.writerow(["Engineering", "", ""])
        # Programs (deduped preferred)
        with (processed / "programs_deduped.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["program_code", "institution_code", "institution_name", "name", "normalized_name", "field_name", "level", "region", "campus"]) 
            w.writerow(["1170114", "1170", "MACHAKOS UNIVERSITY", "BSc Civil Engineering", "BSC CIVIL ENGINEERING", "Engineering", "bachelor", "Eastern", "Main"]) 
        # Point APIs at this processed dir
        os.environ["KUCCPS_PROCESSED_DIR"] = str(processed)
        return tmp

    def test_institutions_endpoint(self):
        tmp = self._setup_processed_dir()
        try:
            resp = self.client.get("/api/etl/institutions", {"q": "machakos"})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertGreaterEqual(data.get("count", 0), 1)
        finally:
            tmp.cleanup()

    def test_fields_endpoint(self):
        tmp = self._setup_processed_dir()
        try:
            resp = self.client.get("/api/etl/fields", {"q": "engineer"})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertGreaterEqual(data.get("count", 0), 1)
        finally:
            tmp.cleanup()

    def test_programs_endpoint(self):
        tmp = self._setup_processed_dir()
        try:
            resp = self.client.get("/api/etl/programs", {"q": "civil", "level": "bachelor"})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertGreaterEqual(data.get("count", 0), 1)
            self.assertTrue(len(data.get("results", [])) <= 20)
        finally:
            tmp.cleanup()

    def test_search_endpoint(self):
        tmp = self._setup_processed_dir()
        try:
            resp = self.client.get("/api/etl/search", {"q": "engineering"})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("programs", data)
            self.assertIn("institutions", data)
            self.assertIn("fields", data)
        finally:
            tmp.cleanup()

    def test_admin_upload_and_process_pages(self):
        tmp = self._setup_processed_dir()
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
