from rest_framework import serializers
from .models import UserProfile


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
