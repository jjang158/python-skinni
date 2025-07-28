from django.urls import path
from .views import ProductRecommendView, SkinAnalysisView

urlpatterns = [
    path('recommend/', ProductRecommendView.as_view(), name='product-recommend'),
    
    path('skin-analysis/', SkinAnalysisView.as_view(), name='skin-analysis'),
]
