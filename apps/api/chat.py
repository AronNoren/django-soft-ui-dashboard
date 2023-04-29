import os
import logging
import requests
from django.http import JsonResponse


logger = logging.getLogger(__name__)

def chat(request):
    try:
        message = "Hi GPT"#request.POST.get('message')
        response = requests.post(
            'https://api.openai.com/v1/completions',#'https://api.openai.com/v1/engines/davinci-codex/completions',
            headers={
                'Authorization': 'Bearer ' + os.getenv('GPT4Key'),
                'Content-Type': 'application/json'
            },
            json={
                'prompt': message,
                'max_tokens': 60,
                'temperature': 0.5,
                'model': 'davinci-codex',
            }
        )
        response.raise_for_status()
        data = response.json()
        return JsonResponse({'reply': data['choices'][0]['text'].strip()})
    except Exception as e:
        logger.error(f"Error during chat: {e}")
        return JsonResponse({'error': 'An error occurred'}, status=500)
