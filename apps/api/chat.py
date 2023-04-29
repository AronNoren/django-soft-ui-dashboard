from django.http import HttpResponse
import openai
import os
import json

api_key = os.getenv('GPT4Key')

def chat(request):
    if api_key is not None and request.method == 'POST':
        openai.api_key = api_key
        user_input = request.POST.get('user_input')

        response = openai.Completion.create(
            engine= 'text-davinci-003',
            prompt= user_input,
            max_tokens=256,
            temperature = 0.5
        )

        # convert the response to string, then parse as json
        response_str = str(response)
        response_json = json.loads(response_str)

        # extract the text from the response
        response_text = response_json["choices"][0]["text"]

        return render(request, 'billing.html', {'response': response_text})
    else:
        return render(request, 'billing.html')
