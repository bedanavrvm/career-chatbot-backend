from django.test import TestCase

from conversations.planner import plan_message, validate_plan_dict
from conversations.tools import get_user_context

from accounts.models import UserProfile, OnboardingProfile
from conversations.models import Session


class TestPlannerSchema(TestCase):
    def test_validate_plan_defaults_and_allowlist(self):
        p = validate_plan_dict({'intent': 'not_a_real_intent', 'entities': {'goal': 'x', 'bad': 1}})
        self.assertEqual(p.intent, 'recommend_programs')
        self.assertIn('goal', p.entities)
        self.assertNotIn('bad', p.entities)
        self.assertIn('k', p.constraints)

    def test_local_planner_basic_recommend(self):
        p = plan_message('I want to do medicine. What programs do you recommend?', provider_override='local')
        self.assertEqual(p.intent, 'recommend_programs')
        self.assertIn('goal', p.entities)


class TestTools(TestCase):
    def test_get_user_context_seeds_grades_from_onboarding(self):
        user = UserProfile.objects.create(uid='u_ctx')
        OnboardingProfile.objects.create(
            user=user,
            education_level='high_school',
            high_school={'subject_grades': {'MAT': 'A-', 'ENG': 'B+'}},
            universal={'careerGoals': ['Medicine']},
            riasec_top=['Investigative', 'Social', 'Realistic'],
        )
        s = Session.objects.create(owner_uid='u_ctx')
        ctx = get_user_context(uid='u_ctx', session_id=str(s.id))
        self.assertEqual(ctx.grades.get('MAT'), 'A-')
        self.assertTrue(ctx.career_goals)
        self.assertIn('Investigative', ctx.traits)
