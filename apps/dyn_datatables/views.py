import json
import math
import random
import string

from django.core.exceptions import ValidationError
from django.db.models import Q
from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from apps.models import Utils

from core.settings import DYNAMIC_DATATB
from django.db.models.fields import DateField

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import base64
import os


# TODO: 404 for wrong page number
def data_table_view(request, **kwargs):
    try:
        model_class = Utils.get_class(DYNAMIC_DATATB, kwargs.get('model_name'))
    except KeyError:
        return render(request, '404.html', status=404)
    headings = [field.name for field in model_class._meta.get_fields()]

    page_number = int(request.GET.get('page', 1))
    search_key = request.GET.get('search', '')
    entries = int(request.GET.get('entries', 10))

    if page_number < 1:
        return render(request, '404.html', status=404)

    filter_options = Q()
    for field in headings:
        filter_options = filter_options | Q(**{field + '__icontains': search_key})
    all_data = model_class.objects.filter(filter_options)
    data = all_data[(page_number - 1) * entries:page_number * entries]
    if all_data.count() != 0 and not 1 <= page_number <= math.ceil(all_data.count() / entries):
        return render(request, '404.html', status=404)
    return render(request, 'index.html', context={
        'model_name': kwargs.get('model_name'),
        'headings': headings,
        'data': [[getattr(record, heading) for heading in headings] for record in data],
        'is_date': [True if type(field) == DateField else False for field in model_class._meta.get_fields()],
        'total_pages': range(1, math.ceil(all_data.count() / entries) + 1),
        'has_prev': False if page_number == 1 else (
            True if all_data.count() != 0 else False),
        'has_next': False if page_number == math.ceil(all_data.count() / entries) else (
            True if all_data.count() != 0 else False),
        'current_page': page_number,
        'entries': entries,
        'search': search_key,
    })


@csrf_exempt
def add_record(request, **kwargs):
    try:
        model_manager = Utils.get_manager(DYNAMIC_DATATB, kwargs.get('model_name'))
    except KeyError:
        return HttpResponse(json.dumps({
            'message': 'this model is not activated or not exist.',
            'success': False
        }), status=400)
    body = json.loads(request.body.decode("utf-8"))
    try:
        thing = model_manager.create(**body)
    except Exception as ve:
        return HttpResponse(json.dumps({
            'detail': str(ve),
            'success': False
        }), status=400)
    return HttpResponse(json.dumps({
        'id': thing.id,
        'message': 'Record Created.',
        'success': True
    }), status=200)


@csrf_exempt
def delete_record(request, **kwargs):
    try:
        model_manager = Utils.get_manager(DYNAMIC_DATATB, kwargs.get('model_name'))
    except KeyError:
        return HttpResponse(json.dumps({
            'message': 'this model is not activated or not exist.',
            'success': False
        }), status=400)
    to_delete_id = kwargs.get('id')
    try:
        to_delete_object = model_manager.get(id=to_delete_id)
    except Exception:
        return HttpResponse(json.dumps({
            'message': 'matching object not found.',
            'success': False
        }), status=404)
    to_delete_object.delete()
    return HttpResponse(json.dumps({
        'message': 'Record Deleted.',
        'success': True
    }), status=200)


@csrf_exempt
def edit_record(request, **kwargs):
    try:
        model_manager = Utils.get_manager(DYNAMIC_DATATB, kwargs.get('model_name'))
    except KeyError:
        return HttpResponse(json.dumps({
            'message': 'this model is not activated or not exist.',
            'success': False
        }), status=400)
    to_update_id = kwargs.get('id')

    body = json.loads(request.body.decode("utf-8"))
    try:
        model_manager.filter(id=to_update_id).update(**body)
    except Exception as ve:
        return HttpResponse(json.dumps({
            'detail': str(ve),
            'success': False
        }), status=400)
    return HttpResponse(json.dumps({
        'message': 'Record Updated.',
        'success': True
    }), status=200)


@csrf_exempt
def export(request, **kwargs):
    try:
        model_class = Utils.get_class(DYNAMIC_DATATB, kwargs.get('model_name'))
    except KeyError:
        return render(request, '404.html', status=404)
    request_body = json.loads(request.body.decode('utf-8'))
    search_key = request_body.get('search', '')
    hidden = request_body.get('hidden_cols', [])
    export_type = request_body.get('type', 'csv')
    filter_options = Q()

    headings = filter(lambda field: field.name not in hidden,
                      [field for field in model_class._meta.get_fields()])
    headings = list(headings)
    for field in headings:
        field_name = field.name
        try:
            filter_options = filter_options | Q(**{field_name + '__icontains': search_key})
        except Exception as _:
            pass

    all_data = model_class.objects.filter(filter_options)
    table_data = []
    for data in all_data:
        this_row = []
        for heading in headings:
            this_row.append(getattr(data, heading.name))
        table_data.append(this_row)

    df = pd.DataFrame(
        table_data,
        columns=tuple(heading.name for heading in headings))
    if export_type == 'pdf':
        base64encoded = get_pdf(df)
    elif export_type == 'xlsx':
        base64encoded = get_excel(df)
    elif export_type == 'csv':
        base64encoded = get_csv(df)
    else:
        base64encoded = 'nothing'

    return HttpResponse(json.dumps({
        'content': base64encoded,
        'file_format': export_type,
        'success': True
    }), status=200)


def get_pdf(data_frame, ):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=data_frame.values, colLabels=data_frame.columns, loc='center',
             colLoc='center', )
    random_file_name = get_random_string(10) + '.pdf'
    pp = PdfPages(random_file_name)
    pp.savefig(fig, bbox_inches='tight')
    pp.close()
    bytess = read_file_and_remove(random_file_name)
    return base64.b64encode(bytess).decode('utf-8')


def get_excel(data_frame, ):
    random_file_name = get_random_string(10) + '.xlsx'

    data_frame.to_excel(random_file_name, index=False, header=True, encoding='utf-8')
    bytess = read_file_and_remove(random_file_name)
    return base64.b64encode(bytess).decode('utf-8')


def get_csv(data_frame, ):
    random_file_name = get_random_string(10) + '.csv'

    data_frame.to_csv(random_file_name, index=False, header=True, encoding='utf-8')
    bytess = read_file_and_remove(random_file_name)
    return base64.b64encode(bytess).decode('utf-8')


def read_file_and_remove(path):
    with open(path, 'rb') as file:
        bytess = file.read()
        file.close()

    # ths file pointer should be closed before removal
    os.remove(path)
    return bytess


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))



from django.http import JsonResponse
import openai
import os
# Set your OpenAI API key
openai.api_key = os.getenv('GPT4Key')
from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def chat(request):
    try:
        print(request)
        if request.method == 'POST':
            data = json.loads(request.body)
            print(data)
            user_message = data.get('message', None)
            print(user_message)
            if user_message:
                response = openai.ChatCompletion.create(
                    model = "gpt-4",#model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are sassy customer support for a crypto website called bjorkfi and you like to use emojis. Bjorkfi is a cryptoindex holding 10 promising coins each month. The coins are based Marketcap, trend and momentum. If the entire market is in a downward trend bjorkfi will instead instruct users to hold USDT during said month. For any further questions regarding strategy you can refer to our website bjorkfi.com. Bjorkfi suggests using Binance to invest and are currently building a python script to automate the trading."},
                        {"role": "user", "content": user_message}
                    ]
                )
                

                ai_message = response['choices'][0]['message']['content']
                print(ai_message)
                return JsonResponse({'message': ai_message})

            else:
                return JsonResponse({'error': 'No message provided'}, status=400)
        else:
            return JsonResponse({'message': 'This is an error message'})
    except Exception as ve:
        return JsonResponse({'detail': str(ve)})

    
    
########################### NEW dev 

#from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
import os
from django.conf import settings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap


#embeddings = OpenAIEmbeddings(openai_api_key = openai.api_key)

@csrf_exempt
def create_db_from_youtube_video_url(embeddings):
    loader = UnstructuredFileLoader(os.path.join(settings.BASE_DIR, 'apps/dyn_datatables/website_data.txt'))
    text = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=500)
    docs = text_splitter.split_documents(text)

    db = FAISS.from_documents(docs, embeddings)
    return db

@csrf_exempt
def get_response_from_query(db, query, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-4", temperature=0.2,openai_api_key=os.getenv('OPENAI_API_KEY'))

    # Template to use for the system message prompt
    template = """
        Du är en hjälpsam assistent som svarar på frågor gällande IF försäkringars olika försäkringar.
        baserat på IF's dokumentation: {docs}
        Dina svar bör vara noggranna och detaljerade. Kom ihåg att bara svara på frågor relaterat till if's försäkringar. Referera också till URLen som hör till infon.
        """
    #Använd bara faktaenlig information från dokumentationen och refferera till hemsidan som finns i dokumnetationen.
    #        dokumentationen är hämtad från if's hemsida och följer formatet
    #        {this is the URL:
    #            url
    #            }
    #            This is the content
    #            {
    #                information
    #            }
    #        Om du inte har tillräckligt med information bör du säga att du inte vet och be användaren kontakta if's support
            
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Svara på följande fråga: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs
@csrf_exempt
def chat_insurance(request):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    db = create_db_from_youtube_video_url(embeddings)

    #query = "Jag vill åka till spanien med min katt. Täcker min försäkring om katten blir sjuk och behöver vård?"
    response, docs = get_response_from_query(db, json.loads(request.body))
    JsonResponse({'message': response})

