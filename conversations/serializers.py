from __future__ import annotations

from rest_framework import serializers


class PostMessageSerializer(serializers.Serializer):
    text = serializers.CharField(allow_blank=False, trim_whitespace=True)
    idempotency_key = serializers.CharField(required=False, allow_blank=True, trim_whitespace=True)
    nlp_provider = serializers.CharField(required=False, allow_blank=True, trim_whitespace=True)
    use_planner = serializers.BooleanField(required=False)

    def validate_text(self, value: str) -> str:
        v = (value or '').strip()
        if not v:
            raise serializers.ValidationError('This field is required.')
        return v

    def validate_nlp_provider(self, value: str) -> str:
        v = (value or '').strip().lower()
        if not v:
            return ''
        if v not in ('local', 'gemini'):
            raise serializers.ValidationError('Invalid provider.')
        return v


class PostProfileSerializer(serializers.Serializer):
    session_id = serializers.UUIDField()
    traits = serializers.DictField(required=False, default=dict)
    grades = serializers.DictField(required=False, default=dict)
    preferences = serializers.DictField(required=False, default=dict)
    version = serializers.CharField(required=False, allow_blank=True, default='v1', trim_whitespace=True)

    def validate(self, attrs):
        for k in ('traits', 'grades', 'preferences'):
            v = attrs.get(k)
            if v is None:
                attrs[k] = {}
            elif not isinstance(v, dict):
                raise serializers.ValidationError({k: ['Expected an object.']})
        return attrs
