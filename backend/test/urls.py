from django.urls import path
from .views import PickerTest

urlpatterns = [
    path('pickerToText/', PickerTest.as_view(), name='picker-to-text'),
]
