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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
        Dina svar bör vara noggranna, faktaenliga och detaljerade. Kom ihåg att bara svara på frågor relaterat till if's försäkringar. Referera också till URLen som hör till infon.
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

import requests
import sys
import numpy as np
import math
import datetime

from binance import Client
from binance.client import Client

def sell(client, amount, symbol="BTCUSDT",logs=False):


    try:
        balance = client.get_asset_balance(asset=symbol[:-4])
        #print(f"Your current {symbol[:-4]} balance is: {balance['free']}")
        if float(balance['free']) >= amount:
            order = client.order_market_sell(symbol=symbol, quantity=amount)
        else:
            print(f"Insufficient {symbol[:-4]} balance to execute the sell order.")
    except Exception as e:
        print(e)

def buy(client, amount, symbol="BTCUSDT", logs=False):
    

    try:
        balance = client.get_asset_balance(asset=symbol[-4:])
        #print(f"Your current {symbol[-4:]} balance is: {balance['free']}")
        order = client.order_market_buy(symbol=symbol, quantity=amount)
        #print("Market buy order sent")
        #print("Order details:")

    except Exception as e:
        print(e)

def place_stop_loss(client, symbol,  quantity, stop_price,stop_price_sell, logs=False):
    try:
        order = client.create_order(
            symbol=symbol,
            side="SELL",
            type="STOP_LOSS_LIMIT",
            quantity=quantity,
            stopPrice=stop_price,
            price=stop_price_sell,
            timeInForce=client.TIME_IN_FORCE_GTC
        )
    except Exception as e:
        print(str(e))
def portfolio(client, logs=False):
    data = client.get_account()
    coins = []
    amounts = []

    for i in data["balances"]:
        if float(i["free"]) +float(i["locked"]) >10**-5:
            coins.append(i["asset"])
            amounts.append(float(i["free"]))
    return coins, amounts

def sell_everything(logs=False):
    api_key = os.getenv('BIN_API_KEY')
    secret_key = os.getenv('BIN_API_KEY')
    client = Client(api_key,secret_key)
    info = client.get_exchange_info()

    old_coins, old_amounts = portfolio(client,logs)
    #print("Old portfolio : ", old_coins, old_amounts)
    total_amounts = np.sum(old_amounts)
    
    coin_lot_sizes = {}
    coin_tick_size = {}
    
    for i in info["symbols"]:
        #print(i)
        if  i["symbol"][:-4] in old_coins and i["symbol"][-4:] == "USDT":
            coin_lot_sizes[i["symbol"][:-4]] = float(i["filters"][1]["stepSize"])
            coin_tick_size[i["symbol"][:-4]] = float(i["filters"][0]["tickSize"])
    for c,a in zip(old_coins, old_amounts):
        if c != "USDT":
            orders = client.get_open_orders(symbol=c+"USDT")
            #print(orders)
            if len(orders) > 0:
                for o in orders:
                    orderId = o["orderId"]
                    result = client.cancel_order(
                        symbol=c+"USDT",
                        orderId=orderId)
            

    old_coins, old_amounts = portfolio(client)

    total_amounts = np.sum(old_amounts)  
    for c,a in zip(old_coins, old_amounts):
        if c != "USDT":
            amount_to_sell = a
            amount_to_sell = float(round(10**5*(math.floor(amount_to_sell/coin_lot_sizes[c])*coin_lot_sizes[c]))/10**5)
            #amount_to_sell = a
            if amount_to_sell > 0:
                sell(client, amount_to_sell,c+"USDT", logs=logs)
                #print("Sold: ", amount_to_sell, " - " ,c+"USDT")
            else:
                #print("Too low quantity to sell")
                pass         
    
def invest(new_coins, logs=False,stop_loss=0.05):
    api_key = os.getenv('BIN_API_KEY')
    secret_key = os.getenv('BIN_API_KEY')
    print("......... Doing Binance Stuff ..........")
        
    client = Client(api_key,secret_key)
    info = client.get_exchange_info()

    old_coins, old_amounts = portfolio(client,logs)
    #print("Old portfolio : ", old_coins, old_amounts)
    total_amounts = np.sum(old_amounts)
    
    coin_lot_sizes = {}
    coin_tick_size = {}
    
    for i in info["symbols"]:
        #print(i)
        if (i["symbol"][:-4] in new_coins or i["symbol"][:-4] in old_coins) and i["symbol"][-4:] == "USDT":
            coin_lot_sizes[i["symbol"][:-4]] = float(i["filters"][1]["stepSize"])
            coin_tick_size[i["symbol"][:-4]] = float(i["filters"][0]["tickSize"])
    successfull_coins = []
    successfull_amounts = []
    for c,a in zip(old_coins, old_amounts):
        if c != "USDT":
            try:
                orders = client.get_open_orders(symbol=c+"USDT")
                successfull_coins.append(c)
                successfull_amounts.append(a)
            except Exception as e:
                print("Coin {} could not be fetched. {}".format(c, e))
            #print(orders)
            if len(orders) > 0:
                for o in orders:
                    orderId = o["orderId"]
                    result = client.cancel_order(
                        symbol=c+"USDT",
                        orderId=orderId)
                    


    old_coins, old_amounts = portfolio(client)

    total_amounts = np.sum(old_amounts)  
    
    for c,a in zip(old_coins, old_amounts):
        if c != "USDT":
            amount_to_sell = a
            amount_to_sell = float(round(10**5*(math.floor(amount_to_sell/coin_lot_sizes[c])*coin_lot_sizes[c]))/10**5)
            #amount_to_sell = a
            if amount_to_sell > 0:
                sell(client, amount_to_sell,c+"USDT", logs=logs)
                #print("Sold: ", amount_to_sell, " - " ,c+"USDT")
            else:
                #print("Too low quantity to sell")
                pass
           
    old_coins, old_amounts = portfolio(client)
    for c,a in zip(old_coins,old_amounts):
        if c=="USDT":
            total_amounts = np.sum(a)*0.99
    
    for c in new_coins:
        #print("-----------------------------------")
        money_per_coin = total_amounts/len(new_coins)
        try:
            coin_price = client.get_avg_price(symbol=c+"USDT")["price"]
            #print("Current price of ", c, " : ", coin_price)
            if coin_price:
                amount_to_buy = money_per_coin / float(coin_price)
                amount_to_buy = float(round(10**5*(math.floor(amount_to_buy/coin_lot_sizes[c])*coin_lot_sizes[c]))/10**5)
                #print("Trying to buy : ", amount_to_buy)
            else:
                amount_to_buy=0
                #print("Did not get a coin price")
            buy(client,amount_to_buy,c+"USDT",logs=logs)
        except Exception as e:
            #print(e)
            if logs:
                print(str(e))
        try:
            balance = client.get_asset_balance(asset=c)
            q = float(balance['free'])
            q = float(round(10**5*(math.floor(q/coin_lot_sizes[c])*coin_lot_sizes[c]))/10**5)

            stop_loss_coin_price = float(coin_price)*(1-stop_loss)
            stop_loss_coin_price = float(round(10**5*(math.floor(stop_loss_coin_price/coin_tick_size[c])*coin_tick_size[c]))/10**5)
            stop_loss_coin_price_sell = float(round(10**5*(math.floor(stop_loss_coin_price*0.99/coin_tick_size[c])*coin_tick_size[c]))/10**5)
            #print(stop_loss_coin_price)
        #print(c, q, coin_price, stop_loss, 1 - stop_loss, coin_price*(1 - stop_loss))
            place_stop_loss(client, c+"USDT", q , float(stop_loss_coin_price),float(stop_loss_coin_price_sell), logs=logs)
        except Exception as e:
            print(e)
        
    print("Done!")
    return 0





def chat_insurance(request):
    coins_to_hold = ['BCHUSDT','COMPUSDT','ETHUSDT','BTCUSDT']
    invest(coins_to_hold,False,0.2)
