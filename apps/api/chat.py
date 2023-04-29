from django.http import JsonResponse
import requests
import os

def chat(request):
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
