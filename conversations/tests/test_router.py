from django.test import TestCase
from unittest.mock import patch

from accounts.models import OnboardingProfile, UserProfile
from catalog.models import Field, Institution, Program
from conversations.compose import compose_response
from conversations.models import Session
from conversations.router import route_turn
from conversations.planner import Plan


class TestRouterPipeline(TestCase):
    def setUp(self):
        self.uid = 'u_router'
        user = UserProfile.objects.create(uid=self.uid)
        OnboardingProfile.objects.create(
            user=user,
            education_level='high_school',
            high_school={'subject_grades': {'MAT': 'A-', 'ENG': 'B+'}},
            universal={'careerGoals': ['Medicine']},
            riasec_top=['Investigative', 'Social', 'Realistic'],
        )
        self.sess = Session.objects.create(owner_uid=self.uid)

        inst_n = Institution.objects.create(code='I_NAI', name='NAIROBI UNI', region='Nairobi', county='Nairobi')
        inst_m = Institution.objects.create(code='I_MOM', name='MOMBASA UNI', region='Coast', county='Mombasa')
        field_med = Field.objects.create(name='Medicine')

        Program.objects.create(
            institution=inst_n,
            field=field_med,
            code='MED1',
            name='BACHELOR OF MEDICINE AND SURGERY (MBCHB)',
            normalized_name='BACHELOR OF MEDICINE AND SURGERY (MBCHB)',
            level='bachelor',
            campus='',
            region='Nairobi',
        )
        Program.objects.create(
            institution=inst_m,
            field=field_med,
            code='MED2',
            name='BACHELOR OF MEDICINE',
            normalized_name='BACHELOR OF MEDICINE',
            level='bachelor',
            campus='',
            region='Mombasa',
        )

    def test_recommend_filter_explain_pipeline(self):
        out1, _nlp1 = route_turn(self.sess, 'I want medicine', uid=self.uid, provider_override='local', tool_budget=2)
        self.assertIsNotNone(out1)
        self.assertEqual(out1.get('type'), 'recommendations')
        items = out1.get('items') or []
        self.assertTrue(any('medicin' in str(it.get('program_name') or '').lower() for it in items))

        out2, _nlp2 = route_turn(self.sess, 'filter by nairobi', uid=self.uid, provider_override='local', tool_budget=1)
        self.assertIsNotNone(out2)
        self.assertEqual(out2.get('type'), 'filtered_results')
        items2 = out2.get('items') or []
        self.assertTrue(all(str(it.get('region') or '').lower().find('nairobi') >= 0 for it in items2))

        out3, _nlp3 = route_turn(self.sess, 'why 1', uid=self.uid, provider_override='local', tool_budget=2)
        self.assertIsNotNone(out3)
        self.assertEqual(out3.get('type'), 'explanation')
        txt = compose_response(out3)
        self.assertIn('Score breakdown', txt)

    def test_tool_budget_blocks_explain(self):
        route_turn(self.sess, 'I want medicine', uid=self.uid, provider_override='local', tool_budget=2)
        out, _ = route_turn(self.sess, 'why 1', uid=self.uid, provider_override='local', tool_budget=1)
        self.assertIsNone(out)

    def test_proposed_actions_allowlist_is_enforced(self):
        fake_plan = Plan(
            intent='recommend_programs',
            entities={'goal': 'medicine'},
            constraints={'k': 10, 'tool_budget': 0},
            clarifying_question='',
            proposed_actions=['recommend_programs', 'drop_table', 'recommend_programs', ''],
        )

        with patch('conversations.router.plan_message', return_value=fake_plan):
            out, meta = route_turn(self.sess, 'anything', uid=self.uid, provider_override='local', tool_budget=0)
            self.assertIsNone(out)
            planner = (meta or {}).get('planner') or {}
            self.assertEqual(planner.get('approved_actions'), ['recommend_programs'])
            self.assertIn('drop_table', planner.get('dropped_actions') or [])
