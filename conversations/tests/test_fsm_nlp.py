from django.test import TestCase
from conversations import nlp
from conversations.fsm import next_turn
from conversations.models import Session, Profile


class TestNLP(TestCase):
    def test_extract_subject_grades(self):
        text = "Math A-; Eng B+; Kiswahili B, Chemistry C+"
        pairs = nlp.extract_subject_grade_pairs(text)
        self.assertEqual(pairs.get('MAT'), 'A-')
        self.assertEqual(pairs.get('ENG'), 'B+')
        self.assertEqual(pairs.get('KIS'), 'B')
        self.assertEqual(pairs.get('CHE'), 'C+')

    def test_analyze_confidence(self):
        res = nlp.analyze("I enjoy coding. Math A- and Physics B.")
        self.assertIn('grades', res)
        self.assertIn('traits', res)
        self.assertIn('intents', res)
        self.assertGreaterEqual(res.get('confidence', 0), 0.2)


class TestFSM(TestCase):
    def test_fsm_happy_path(self):
        s = Session.objects.create()
        s.ensure_ttl()
        s.save()

        # 1) Greeting -> ask for grades
        t1 = next_turn(s, "Hi")
        self.assertEqual(t1.next_state, 'collect_grades')
        self.assertIn('grades', t1.reply.lower())

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
        self.assertEqual(t3.next_state, 'summarize')
        self.assertIn("here's what i have", t3.reply.lower())

        # Profile updated with grades
        prof = Profile.objects.get(session=s)
        self.assertIn('MAT', prof.grades)
