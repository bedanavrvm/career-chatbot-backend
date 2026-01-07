from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response

from utils.drf_auth import FirebaseAuthentication, IsFirebaseAuthenticated


@api_view(['GET'])
def health(_request):
    return Response({"status": "ok"})


@api_view(['GET'])
@authentication_classes([FirebaseAuthentication])
@permission_classes([IsFirebaseAuthenticated])
def secure_ping(request):
    return Response({"status": "ok", "uid": request.user.uid})
