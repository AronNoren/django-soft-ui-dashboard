from django.urls import re_path, path
from django.views.decorators.csrf import csrf_exempt

from apps.api.views import BookView,ChatView
from apps.api.chat import chat
urlpatterns = [

	re_path("books/((?P<pk>\d+)/)?", csrf_exempt(BookView.as_view())),
        path("", chat, name="chat"),

]
