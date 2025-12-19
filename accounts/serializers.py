from rest_framework import serializers
from .models import UserProfile, OnboardingProfile


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = [
            'uid',
            'email',
            'display_name',
            'photo_url',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['created_at', 'updated_at']


class OnboardingProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = OnboardingProfile
        fields = [
            'education_level',
            'universal',
            'high_school',
            'college',
            'riasec_answers',
            'riasec_scores',
            'riasec_top',
            'strengths',
            'skills',
            'work_style',
            'lifestyle',
            'preferences',
            'status',
            'version',
            'created_at',
            'updated_at',
        ]
        read_only_fields = ['riasec_scores', 'riasec_top', 'created_at', 'updated_at']
