from rest_framework import serializers
from .models import ProductMgmt

class ProductMgmtSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProductMgmt
        fields = '__all__'
