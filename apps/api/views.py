from http import HTTPStatus
from django.http import Http404
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.generics import get_object_or_404
from rest_framework.permissions import IsAuthenticatedOrReadOnly


from apps.api.serializers import BookSerializer
from apps.models import Book

from http import HTTPStatus
from django.http import Http404, JsonResponse
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.generics import get_object_or_404
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from apps.api.serializers import BookSerializer
from apps.models import Book
import os
import logging
import requests

logger = logging.getLogger(__name__)

class BookView(APIView):

    permission_classes = (IsAuthenticatedOrReadOnly,)

    def post(self, request):
        serializer = BookSerializer(data=request.POST)
        if not serializer.is_valid():
            return Response(data={
                **serializer.errors,
                'success': False
            }, status=HTTPStatus.BAD_REQUEST)
        serializer.save()
        return Response(data={
            'message': 'Record Created.',
            'success': True
        }, status=HTTPStatus.OK)

    def get(self, request, pk=None):
        if not pk:
            return Response({
                'data': [BookSerializer(instance=obj).data for obj in Book.objects.all()],
                'success': True
            }, status=HTTPStatus.OK)
        try:
            obj = get_object_or_404(Book, pk=pk)
        except Http404:
            return Response(data={
                'message': 'object with given id not found.',
                'success': False
            }, status=HTTPStatus.NOT_FOUND)
        return Response({
            'data': BookSerializer(instance=obj).data,
            'success': True
        }, status=HTTPStatus.OK)

    def put(self, request, pk):
        try:
            obj = get_object_or_404(Book, pk=pk)
        except Http404:
            return Response(data={
                'message': 'object with given id not found.',
                'success': False
            }, status=HTTPStatus.NOT_FOUND)
        serializer = BookSerializer(instance=obj, data=request.POST, partial=True)
        if not serializer.is_valid():
            return Response(data={
                **serializer.errors,
                'success': False
            }, status=HTTPStatus.BAD_REQUEST)
        serializer.save()
        return Response(data={
            'message': 'Record Updated.',
            'success': True
        }, status=HTTPStatus.OK)

    def delete(self, request, pk):
        try:
            obj = get_object_or_404(Book, pk=pk)
        except Http404:
            return Response(data={
                'message': 'object with given id not found.',
                'success': False
            }, status=HTTPStatus.NOT_FOUND)
        obj.delete()
        return Response(data={
            'message': 'Record Deleted.',
            'success': True
        }, status=HTTPStatus.OK)



class ChatView(APIView):
    
    def post(self, request):
        try:
            message = request.POST.get('message')
            response = requests.post(
                'https://api.openai.com/v1/engines/davinci-codex/completions',
                headers={
                    'Authorization': 'Bearer ' + os.getenv('GPT4Key'),
                    'Content-Type': 'application/json'
                },
                json={
                    'prompt': message,
                    'max_tokens': 60,
                    'temperature': 0.5,
                }
            )
            response.raise_for_status()
            data = response.json()
            return JsonResponse({'reply': data['choices'][0]['text'].strip()})
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            return Response({'error': 'An error occurred'}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
