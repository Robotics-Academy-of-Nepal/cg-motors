from django.urls import path
from . import views

urlpatterns = [
    path('response/', views.chatbot_response, name='chatbot_response'),  # Route for chatbot API
]
