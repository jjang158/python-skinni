# backend/recommend/models.py

from django.db import models

class CompanyInfo(models.Model):
    id = models.CharField(primary_key=True, max_length=50)
    name = models.CharField(max_length=255)
    url = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        db_table = "company_info"

class ProductMgmt(models.Model):
    id = models.CharField(max_length=50, primary_key=True)
    company = models.ForeignKey(CompanyInfo, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    price = models.DecimalField(max_digits=10, decimal_places=0, default=0)
    description = models.TextField(null=True, blank=True)
    image_path = models.CharField(max_length=255, null=True, blank=True)
    commerce_url = models.CharField(max_length=1000, null=True, blank=True)
    wrinkle = models.IntegerField(default=0)
    pore = models.IntegerField(default=0)
    elasticity = models.IntegerField(default=0)
    moisture = models.IntegerField(default=0)
    crt_user = models.CharField(max_length=100, null=True, blank=True)
    crtdt = models.DateTimeField(auto_now_add=True)
    updt_user = models.CharField(max_length=100, null=True, blank=True)
    updtdt = models.DateTimeField(auto_now=True)

    @property
    def image_url(self):
        return f"http://43.202.92.248{self.image_path}"

    class Meta:
        db_table = "product_mgmt"
