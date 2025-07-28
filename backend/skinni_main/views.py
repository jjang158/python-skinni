from rest_framework.views import APIView
from rest_framework.response import Response
from django.db.models import F
from .models import ProductMgmt
from .serializers import ProductMgmtSerializer


class SkinAnalysisView(APIView):
    def post(self, request):
        sample_data = {
            "status": 200,
            "result":{
                "success": True,
                "model_version": "최종",
                "parts": {
                    "forehead": {"moisture": 58.29, "elasticity": 53.09},
                    "l_cheek": {"moisture": 53.86, "elasticity": 55.44, "pore": 990.06},
                    "r_cheek": {"moisture": 52.94, "elasticity": 55.86, "pore": 1098.26}, 
                    "l_perocular": {"wrinkle": 26.34},
                    "r_perocular": {"wrinkle": 26.65},
                    "chin": {"moisture": 58.44, "elasticity": 45.44}
                },
                "averages": {
                    "moisture": 55.89,
                    "elasticity": 52.46,
                    "wrinkle": 26.49,
                    "pore": 49.58
                }
            }
        }
        return Response(sample_data)



class ProductRecommendView(APIView):

    def post(self, request):
        # 테스트용 하드코딩된 응답
        sample_data = {
            'status' : 200,
            'result' : [{
                "id": "prod001",
                "company_id": "comp001",
                "name": "리페어 크림",
                "price": "32000",
                "description": "피부 장벽을 강화하는 고보습 크림",
                "commerce_url": "https://shop.example.com/product/repair-cream",
                "wrinkle": 4,
                "pore": 2,
                "elasticity": 5,
                "moisture": 5
            },
            {
                "id": "prod002",
                "company_id": "comp001",
                "name": "수분 토너",
                "price": "18000",
                "description": "촉촉하게 수분을 공급하는 데일리 토너",
                "commerce_url": "https://shop.example.com/product/moisture-toner",
                "wrinkle": 2,
                "pore": 3,
                "elasticity": 3,
                "moisture": 4
            }]
        }
        return Response(sample_data)

    # def post(self, request):
    #     scores = {
    #         'wrinkle': int(request.data.get('wrinkle', 0)),
    #         'pore': int(request.data.get('pore', 0)),
    #         'elasticity': int(request.data.get('elasticity', 0)),
    #         'moisture': int(request.data.get('moisture', 0)),
    #     }

    #     lowest = sorted(scores.items(), key=lambda x: x[1])[:2]
    #     lowest_features = [feature for feature, _ in lowest]

    #     recommended = []
    #     for feature in lowest_features:
    #         products = ProductMgmt.objects.order_by(F(feature).desc())[:3]
    #         recommended += list(products)

    #     # ID 기준 중복 제거
    #     unique_recommendations = {p.id: p for p in recommended}.values()

    #     serializer = ProductMgmtSerializer(unique_recommendations, many=True)
    #     return Response(serializer.data)
