from django.db import models


class UserProfile(models.Model):
    uid = models.CharField(max_length=128, unique=True)
    email = models.EmailField(blank=True, null=True)
    display_name = models.CharField(max_length=255, blank=True, null=True)
    photo_url = models.URLField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.uid


class OnboardingProfile(models.Model):
    """Stores the user's multi-section onboarding data and computed scores.

    Sections captured (all optional to allow progressive save):
    - Universal (personal info, interests/personality, preferences)
    - High school branch (school, subjects, grades, extracurriculars)
    - College/graduate branch (institution, qualification, field of study, year, work status)
    - RIASEC answers and computed scores/top types
    - Strengths/skills/competencies
    - Lifestyle & work preferences

    Inline computation helpers let the backend compute RIASEC totals from Likert choices
    (Agree=2, Neutral=1, Disagree=0) and derive top codes for dashboard display.
    """

    class EducationLevel(models.TextChoices):
        HIGH_SCHOOL = "high_school", "High School"
        COLLEGE_STUDENT = "college_student", "College/University Student"
        COLLEGE_GRAD = "college_graduate", "College/University Graduate"

    user = models.OneToOneField(UserProfile, on_delete=models.CASCADE, related_name="onboarding")
    education_level = models.CharField(max_length=32, choices=EducationLevel.choices, blank=True)

    # Section A: Universal
    universal = models.JSONField(default=dict, blank=True)

    # Branch: High School
    high_school = models.JSONField(default=dict, blank=True)

    # Branch: College/University (student or graduate)
    college = models.JSONField(default=dict, blank=True)

    # Section 2: Career Interest & Personality (RIASEC)
    riasec_answers = models.JSONField(default=dict, blank=True)  # e.g., {"Realistic": [2,1,2,...], ...}
    riasec_scores = models.JSONField(default=dict, blank=True)   # e.g., {"Realistic": 8, ...}
    riasec_top = models.JSONField(default=list, blank=True)      # e.g., ["Artistic","Investigative","Social"]

    # Section 3: Strengths, Skills & Competencies
    strengths = models.JSONField(default=dict, blank=True)
    skills = models.JSONField(default=list, blank=True)
    work_style = models.JSONField(default=dict, blank=True)

    # Section 4: Lifestyle & Work Preferences
    lifestyle = models.JSONField(default=dict, blank=True)
    preferences = models.JSONField(default=dict, blank=True)

    status = models.CharField(max_length=32, default="incomplete", blank=True)
    version = models.CharField(max_length=16, default="v1", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"onboarding:{self.user.uid}"

    # ---- Inline computation helpers ----
    def compute_riasec(self) -> None:
        """Compute RIASEC scores/top types from riasec_answers.

        Answers format supports a dict of lists: {CodeName: [2,1,0,...], ...}.
        We sum each list and sort descending to derive top types (top 3 by default).
        """
        ans = self.riasec_answers or {}
        scores = {}
        for code, arr in (ans.items() if isinstance(ans, dict) else []):
            try:
                total = sum(int(x) for x in (arr or []) if isinstance(x, (int, float)))
            except Exception:
                total = 0
            scores[str(code)] = int(total)
        # Persist ordered top 3
        tops = [k for k, _v in sorted(scores.items(), key=lambda kv: -kv[1])[:3]] if scores else []
        self.riasec_scores = scores
        self.riasec_top = tops
