from rest_framework import serializers
from .models import ProductMgmt, CompanyInfo

class CompanyInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = CompanyInfo
        fields = ['name', 'url']

class ProductMgmtSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()
    company = CompanyInfoSerializer(read_only=True)

    class Meta:
        model = ProductMgmt
        fields = ['id', 'name', 'price','description','wrinkle', 'pore', 'elasticity', 'moisture', 'image_url','commerce_url', 'company']

    def get_image_url(self, obj):
        request = self.context.get('request')
        if request:
            return request.build_absolute_uri(obj.image_path)
        return obj.image_path


class RecommendInputSerializer(serializers.Serializer):
    wrinkle = serializers.IntegerField()
    pore = serializers.IntegerField()
    elasticity = serializers.IntegerField()
    moisture = serializers.IntegerField()