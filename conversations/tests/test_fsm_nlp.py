from django.test import TestCase
from conversations import nlp
from conversations.fsm import next_turn
from conversations.models import Session, Profile
from conversations.providers import gemini_provider
from catalog.models import Institution, Field, Program, ProgramRequirementGroup, ProgramRequirementOption, Subject
from accounts.models import UserProfile, OnboardingProfile


class TestNLP(TestCase):
    def test_extract_subject_grades(self):
        text = "Math A-; Eng B+; Kiswahili B, Chemistry C+"
        pairs = nlp.extract_subject_grade_pairs(text)
        self.assertEqual(pairs.get('MAT'), 'A-')
        self.assertEqual(pairs.get('ENG'), 'B+')
        self.assertEqual(pairs.get('KIS'), 'B')
        self.assertEqual(pairs.get('CHE'), 'C+')

    def test_catalog_lookup_intent_for_programs_on_field(self):
        res = nlp.analyze("Are there programs on music")
        intents = res.get('intents') or []
        self.assertIn('catalog_lookup', intents)
        self.assertNotIn('greeting', intents)

    def test_no_false_greeting_from_substring(self):
        res = nlp.analyze("Which universities offer music")
        intents = res.get('intents') or []
        self.assertNotIn('greeting', intents)

    def test_programs_near_me_intent(self):
        res = nlp.analyze("Which program is offered close to my home location?")
        intents = res.get('intents') or []
        self.assertIn('programs_near_me', intents)

    def test_programs_near_me_intent_local_nlp(self):
        res = nlp.analyze("Which program is offered close to my home location?", provider_override='local')
        intents = res.get('intents') or []
        self.assertIn('programs_near_me', intents)

    def test_analyze_confidence(self):
        res = nlp.analyze("I enjoy coding. Math A- and Physics B.")
        self.assertIn('grades', res)
        self.assertIn('traits', res)
        self.assertIn('intents', res)
        self.assertGreaterEqual(res.get('confidence', 0), 0.2)

    def test_new_intents_explain_qualify_career_paths(self):
        res1 = nlp.analyze("Why?")
        self.assertIn('explain', res1.get('intents') or [])

        res2 = nlp.analyze("Which of these do I qualify for?")
        self.assertIn('qualify', res2.get('intents') or [])

        res3 = nlp.analyze("So what are the possible career paths for an aspiring musician?")
        self.assertIn('career_paths', res3.get('intents') or [])

    def test_gemini_plain_text_sanitizer_removes_asterisks(self):
        raw = "* One\n* Two\n\n**Bold** and `code`"
        out = gemini_provider._sanitize_plain_text(raw)
        self.assertNotIn('*', out)
        self.assertNotIn('**', out)
        self.assertIn('1. One', out)
        self.assertIn('2. Two', out)


class TestFSM(TestCase):
    def _seed_catalog_program(self):
        inst = Institution.objects.create(code='TST1', name='TEST UNIVERSITY', region='Nairobi', county='Nairobi')
        field = Field.objects.create(name='Arts')

        p = Program.objects.create(
            institution=inst,
            field=field,
            code='TST101',
            name='BACHELOR OF MUSIC',
            normalized_name='BACHELOR OF MUSIC',
            level='bachelor',
            campus='',
            region='Nairobi',
        )

        subj_eng = Subject.objects.create(code='ENG', name='English')
        g1 = ProgramRequirementGroup.objects.create(program=p, name='Group 1', pick=1, order=0)
        ProgramRequirementOption.objects.create(group=g1, subject=subj_eng, subject_code='ENG', min_grade='C', order=0)
        return p

    def test_qualify_specific_program_without_last_recommendations_returns_reasons_and_alternatives(self):
        inst = Institution.objects.create(code='MED2', name='MED TEST UNIVERSITY', region='Nairobi', county='Nairobi')

        field_med = Field.objects.create(name='Medicine')
        p_med = Program.objects.create(
            institution=inst,
            field=field_med,
            code='MED100',
            name='BACHELOR OF MEDICINE AND BACHELOR OF SURGERY (MBCHB)',
            normalized_name='BACHELOR OF MEDICINE AND BACHELOR OF SURGERY (MBCHB)',
            level='bachelor',
            campus='',
            region='Nairobi',
        )

        subj_che = Subject.objects.create(code='CHE', name='Chemistry')
        g_med = ProgramRequirementGroup.objects.create(program=p_med, name='Group 1', pick=1, order=0)
        ProgramRequirementOption.objects.create(group=g_med, subject=subj_che, subject_code='CHE', min_grade='B', order=0)

        field_nur = Field.objects.create(name='Nursing')
        p_nur = Program.objects.create(
            institution=inst,
            field=field_nur,
            code='NUR100',
            name='BACHELOR OF SCIENCE IN NURSING',
            normalized_name='BACHELOR OF SCIENCE IN NURSING',
            level='bachelor',
            campus='',
            region='Nairobi',
        )
        subj_eng = Subject.objects.create(code='ENG', name='English')
        g_nur = ProgramRequirementGroup.objects.create(program=p_nur, name='Group 1', pick=1, order=0)
        ProgramRequirementOption.objects.create(group=g_nur, subject=subj_eng, subject_code='ENG', min_grade='C', order=0)

        s = Session.objects.create(owner_uid='u_med', fsm_state='greeting', slots={})
        Profile.objects.create(
            session=s,
            traits={'Social': 1.0, 'Investigative': 0.5},
            grades={'MAT': 'A-', 'BIO': 'B-', 'ENG': 'B+'},
            preferences={},
        )

        t = next_turn(s, 'Can I do medicine?', provider_override='local')
        self.assertIn('do not qualify', t.reply.lower())
        self.assertIn('common missing requirements', t.reply.lower())
        self.assertIn('che', t.reply.lower())
        self.assertIn('programs you qualify for right now', t.reply.lower())
        self.assertIn('nursing', t.reply.lower())

    def test_fsm_happy_path(self):
        s = Session.objects.create()
        s.ensure_ttl()
        s.save()

        # 1) Greeting -> general help (no immediate grade prompt)
        t1 = next_turn(s, "Hi")
        self.assertEqual(t1.next_state, 'greeting')
        self.assertTrue(any(k in t1.reply.lower() for k in ['help', 'recommend', 'program']))

        # Apply transition
        s.fsm_state = t1.next_state
        s.slots = t1.slots
        s.save()

        # 2) Provide grades -> ask for interests
        t2 = next_turn(s, "Math A-, English B+")
        self.assertEqual(t2.next_state, 'collect_interests')
        self.assertIn('enjoy', t2.reply.lower())

        s.fsm_state = t2.next_state
        s.slots = t2.slots
        s.save()

        # 3) Provide interests -> summarize
        t3 = next_turn(s, "I enjoy coding and design")
        self.assertEqual(t3.next_state, 'recommend')
        self.assertIn("here's what i have", t3.reply.lower())

        # Profile updated with grades
        prof = Profile.objects.get(session=s)
        self.assertIn('MAT', prof.grades)

    def test_no_collect_grades_when_onboarding_has_grades_and_goal_drives_recs(self):
        user = UserProfile.objects.create(uid='u_onb')
        OnboardingProfile.objects.create(
            user=user,
            education_level='high_school',
            high_school={
                'subject_grades': {
                    'MAT': 'A-',
                    'ENG': 'B+',
                    'BIO': 'A',
                    'CHE': 'A',
                }
            },
        )

        inst = Institution.objects.create(code='MED1', name='MED UNIVERSITY', region='Nairobi', county='Nairobi')
        field_med = Field.objects.create(name='Medicine')
        Program.objects.create(
            institution=inst,
            field=field_med,
            code='MED001',
            name='BACHELOR OF MEDICINE AND BACHELOR OF SURGERY (MBCHB)',
            normalized_name='BACHELOR OF MEDICINE AND BACHELOR OF SURGERY (MBCHB)',
            level='bachelor',
            campus='',
            region='Nairobi',
        )

        field_other = Field.objects.create(name='Education')
        Program.objects.create(
            institution=inst,
            field=field_other,
            code='EDU001',
            name='BACHELOR OF EDUCATION (ARTS - BUSINESS STUDIES)',
            normalized_name='BACHELOR OF EDUCATION (ARTS - BUSINESS STUDIES)',
            level='bachelor',
            campus='',
            region='Nairobi',
        )

        s = Session.objects.create(owner_uid='u_onb', fsm_state='greeting', slots={})
        t = next_turn(s, 'I want to do medicine. What programs do you recommend?')

        self.assertEqual(t.next_state, 'recommend')
        self.assertNotIn('share your kcse grades', t.reply.lower())
        self.assertIn('mbchb', t.reply.lower())

    def test_recommend_without_grades_when_traits_exist(self):
        s = Session.objects.create(owner_uid='u1', fsm_state='greeting', slots={})
        Profile.objects.create(session=s, traits={'Artistic': 1.0}, grades={}, preferences={})
        t = next_turn(s, "what is my recommended career path?")
        self.assertNotEqual(t.next_state, 'greeting')

    def test_programs_near_me_uses_saved_region(self):
        s = Session.objects.create(owner_uid='u1', fsm_state='greeting', slots={})
        Profile.objects.create(session=s, traits={'Artistic': 1.0}, grades={}, preferences={'region': 'Nairobi'})
        t = next_turn(s, "Which program is offered close to my home location?")
        self.assertIn('home', t.reply.lower())

        # Should not fall back to generic greeting when the intent is clear
        self.assertNotEqual(t.next_state, 'greeting')

    def test_fsm_career_paths_musician(self):
        s = Session.objects.create(owner_uid='u1', fsm_state='greeting', slots={})
        Profile.objects.create(session=s, traits={'Artistic': 1.0}, grades={}, preferences={})
        t = next_turn(s, 'So what are the possible career paths for an aspiring musician?')
        self.assertIn('career paths', t.reply.lower())
        self.assertTrue(any(tok in t.reply.lower() for tok in ['music', 'musician']))

    def test_fsm_explain_and_qualify_followups(self):
        self._seed_catalog_program()
        s = Session.objects.create(owner_uid='u1', fsm_state='greeting', slots={})
        Profile.objects.create(session=s, traits={'Artistic': 1.0}, grades={'ENG': 'B+'}, preferences={})

        t1 = next_turn(s, 'recommend')
        self.assertTrue(any(tok in t1.reply.lower() for tok in ['recommend', 'following program']))
        self.assertTrue(isinstance(t1.slots.get('last_recommendations'), list))
        self.assertGreater(len(t1.slots.get('last_recommendations') or []), 0)

        s.fsm_state = t1.next_state
        s.slots = t1.slots
        s.save()

        t2 = next_turn(s, 'Why?')
        self.assertIn('why these were recommended', t2.reply.lower())

        s.fsm_state = t2.next_state
        s.slots = t2.slots
        s.save()

        t3 = next_turn(s, 'Which of these do I qualify for?')
        self.assertTrue(any(tok in t3.reply.lower() for tok in ['qualify', 'eligible', 'eligibility']))
