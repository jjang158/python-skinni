# backend/recommend/models.py

from django.db import models

class ProductMgmt(models.Model):
    id = models.CharField(max_length=50, primary_key=True)
    company_id = models.CharField(max_length=50)
    name = models.CharField(max_length=255)
    price = models.DecimalField(max_digits=10, decimal_places=0, default=0)
    description = models.TextField(null=True, blank=True)
    image_path = models.CharField(max_length=255, null=True, blank=True)
    commerce_url = models.CharField(max_length=255, null=True, blank=True)
    wrinkle = models.IntegerField(default=0)
    pore = models.IntegerField(default=0)
    elasticity = models.IntegerField(default=0)
    moisture = models.IntegerField(default=0)
    crt_user = models.CharField(max_length=100, null=True, blank=True)
    crtdt = models.DateTimeField(auto_now_add=True)
    updt_user = models.CharField(max_length=100, null=True, blank=True)
    updtdt = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "product_mgmt"
