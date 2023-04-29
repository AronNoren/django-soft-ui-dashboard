import os
import logging
import requests
from django.http import JsonResponse
import openai
from django.shortcuts import render
logger = logging.getLogger(__name__)
api_key = os.getenv('GPT4Key')
def chat(request):
    chatresponse = None

    if api_key is not None and request.method == 'POST':
        message = "Hi GPT"#request.POST.get('message')
        openai.api_key = api_key
        user_input = request.POST.get('user_input')
        response = openai.Completion.create(
            engine= 'text-davinci-003',
            prompt= user_input,
            max_tokens=256,
            temperature = 0.5

        )
        print(response)
        return render((request,'billing.html',{}))

