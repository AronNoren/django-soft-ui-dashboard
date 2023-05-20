from django.contrib import admin
from django.urls import path
from .views import data_table_view, add_record, delete_record, edit_record, export, chat, chat_insurance

urlpatterns = [
    path('datatb/<str:model_name>/', data_table_view),
    path('datatb/<str:model_name>/add/', add_record),
    path('datatb/<str:model_name>/edit/<int:id>/', edit_record),
    path('datatb/<str:model_name>/delete/<int:id>/', delete_record),
    path('datatb/<str:model_name>/export/', export),
    path('chat1/', chat),
    path('chat_insurance/', chat_insurance),
]
