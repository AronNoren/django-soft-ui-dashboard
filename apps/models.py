import datetime

from django.db import models
from rest_framework import serializers
import sys
import inspect

'''
Leave this Helper Class in the TOP of the file
'''
class Utils:
    @staticmethod
    def get_class(config, name: str) -> models.Model:
        return Utils.model_name_to_class(config[name])

    @staticmethod
    def get_manager(config, name: str) -> models.Manager:
        return Utils.get_class(config, name).objects

    @staticmethod
    def get_serializer(config, name: str):
        class Serializer(serializers.ModelSerializer):
            class Meta:
                model = Utils.get_class(config, name)
                fields = '__all__'

        return Serializer

    @staticmethod
    def model_name_to_class(name: str):
        all_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
        for cls in all_classes:
            if cls[0] == name:
                return cls[1]
        # we are confident that never returns None
        return None

'''
Add your models below
'''

class Book(models.Model):
    class Meta:
        app_label = 'dyn_datatables'

    coin = models.CharField(max_length=100)
    returns = models.CharField(max_length=100)
    cumprod = models.CharField(max_length=100)
    
class Portfolio(models.Model):
    class Meta:
        app_label = 'dyn_datatables'

    token = models.CharField(max_length=100)
    buy = models.CharField(max_length=100)
    weight = models.CharField(max_length=100)
    amount = models.CharField(max_length=100)
    current = models.CharField(max_length=100)
    
class Returns(models.Model):
    class Meta:
        app_label = 'dyn_datatables'

    date = models.CharField(max_length=100)
    dailyreturnLucy = models.CharField(max_length=100)
    cumdailyreturnLucy = models.CharField(max_length=100)
    dailyreturnLucyhold = models.CharField(max_length=100)
    cumdailyreturnLucyhold = models.CharField(max_length=100)
    dailyreturnBTC = models.CharField(max_length=100)
    cumdailyreturnBTC = models.CharField(max_length=100)
    dailyreturnBTChold = models.CharField(max_length=100)
    cumdailyreturnBTChold = models.CharField(max_length=100)

    

