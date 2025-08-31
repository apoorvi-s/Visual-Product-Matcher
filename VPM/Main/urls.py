from django.contrib import admin
from django.urls import path
from Main import views

urlpatterns = [
    path('', views.upload, name='Upload'),
    path('find_similar/', views.find_similar, name='find_similar'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact')
]
