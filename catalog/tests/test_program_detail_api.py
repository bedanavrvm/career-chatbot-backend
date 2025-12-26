from decimal import Decimal

from django.test import TestCase, Client


class CatalogProgramDetailApiTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_program_detail_endpoint(self):
        from catalog.models import (
            Institution,
            Field,
            Program,
            YearlyCutoff,
            ProgramCost,
            Subject,
            ProgramRequirementGroup,
            ProgramRequirementOption,
        )

        inst = Institution.objects.create(
            code="I1",
            name="TEST UNIVERSITY",
            region="Nairobi",
            county="Nairobi",
            website="https://example.edu",
        )
        field = Field.objects.create(name="Engineering")
        prog = Program.objects.create(
            institution=inst,
            field=field,
            code="P1",
            name="BSc Civil Engineering",
            normalized_name="BSC CIVIL ENGINEERING",
            level="bachelor",
            campus="Main",
            region="Nairobi",
            duration_years=Decimal("4.0"),
            award="BSc",
            mode="Full-time",
            subject_requirements={},
            metadata={},
        )

        YearlyCutoff.objects.create(
            program=prog,
            year=2024,
            cutoff=Decimal("30.500"),
            capacity=100,
            notes="",
        )
        YearlyCutoff.objects.create(
            program=prog,
            year=2023,
            cutoff=Decimal("29.000"),
            capacity=None,
            notes="",
        )

        ProgramCost.objects.create(
            program=prog,
            program_code="P1",
            institution_name="TEST UNIVERSITY",
            program_name="BSc Civil Engineering",
            amount=Decimal("1000.00"),
            currency="KES",
            source_id="test",
            raw_cost="",
            metadata={},
        )

        subj = Subject.objects.create(code="MAT", name="Mathematics")
        g = ProgramRequirementGroup.objects.create(program=prog, name="Group 1", pick=1, order=1)
        ProgramRequirementOption.objects.create(
            group=g,
            subject=subj,
            subject_code="MAT",
            min_grade="B+",
            order=1,
            metadata={},
        )

        resp = self.client.get(f"/api/catalog/programs/{prog.id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data.get("id"), prog.id)
        self.assertEqual((data.get("institution") or {}).get("website"), "https://example.edu")

        cutoffs = data.get("cutoffs") or []
        self.assertEqual(len(cutoffs), 2)
        self.assertTrue(any(int(c.get("year") or 0) == 2024 for c in cutoffs))

        costs = data.get("costs") or []
        self.assertGreaterEqual(len(costs), 1)

        groups = data.get("requirement_groups") or []
        self.assertGreaterEqual(len(groups), 1)
        opts = (groups[0].get("options") or [])
        self.assertGreaterEqual(len(opts), 1)
        self.assertEqual((opts[0].get("subject_code") or "").upper(), "MAT")
        self.assertEqual(opts[0].get("min_grade"), "B+")
