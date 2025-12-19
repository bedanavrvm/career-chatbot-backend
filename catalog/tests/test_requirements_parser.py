import os
import sys
import importlib.util
from pathlib import Path
from django.test import TestCase


class RequirementsParserTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Dynamically import the ETL module to access _parse_requirements
        backend_dir = Path(__file__).resolve().parents[2]  # .../DELPHINE/backend
        etl_path = backend_dir / "scripts" / "etl" / "kuccps" / "etl.py"
        spec = importlib.util.spec_from_file_location("kuccps_etl", str(etl_path))
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        # Register module to ensure dataclasses and annotations resolve __module__ properly
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)  # type: ignore
        cls.etl = module

    def test_parse_requirements_semicolons_and_or(self):
        parse = self.etl._parse_requirements
        cells = [
            "ENG (101) : B+ / MAT (121) : B",
            "BIO (231) or CHEM (233): B-; PHY (232): C+",
        ]
        out = parse(cells)
        # Should have groups for alternatives and required entries
        self.assertIn("groups", out)
        self.assertTrue(any("options" in g for g in out.get("groups", [])))
        # Basic sanity on parsed items
        all_opts = []
        for g in out.get("groups", []):
            all_opts.extend(g.get("options", []))
        reqs = out.get("required", [])
        # Expect at least 3 unique subject codes present
        codes = {o.get("code") for o in all_opts} | {r.get("code") for r in reqs}
        self.assertGreaterEqual(len({c for c in codes if c}), 3)
