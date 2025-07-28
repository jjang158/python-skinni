from rest_framework.views import APIView
from rest_framework.response import Response
from django.db.models import F
from .models import ProductMgmt
from .serializers import ProductMgmtSerializer, RecommendInputSerializer
from .run_model import get_model_info, analyze_skin_image

# 피부 분석 API
class SkinAnalysisView(APIView):
    def post(self, request):
        result_obj = {}
        
        try:
            print("피부 분석 모델 테스트")
            info = get_model_info()
            print(f"모델: {info['model_type']}")
            print(f"버전: {info['model_version']}")
            print(f"분석 부위: {info['total_parts']}개")
            print(f"특화: {info['specialization']}")
            
            # 예시 base64 이미지 데이터
#             base64_image = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxAAPwCdABmX/9k="
#             result = analyze_skin_image(base64_image)
            result = analyze_skin_image(request.data.get('file'))
            
            if result['success']:
                print(f"\n모델 버전: {result['model_version']}")
                print("\n=== 부위별 Q0/Ra 분석 ===")
                for part_name, part_data in result['parts'].items():
                    print(f"{part_name}:")
                    for param_name, value in part_data.items():
                        print(f"  {param_name}: {value:.2f}")
                
                print(f"\n=== 총 평균 ===")
                for param_name, avg_value in result['averages'].items():
                    print(f"{param_name}: {avg_value:.2f}")
                    
                result_obj = {
                    "status": 200,
                    "message": "success",
                    "result": result
                }
            else:
                print(f"analyze_skin_image fail: {result['error']}")
                result_obj = {
                    "status": 500,
                    "message": result['error'],
                    "result": {}
                }
                
        except Exception as e:
            print(f"Runtime Exception: {e}")
            result_obj = {
                "status": 500,
                "message": str(e),
                "result": {}
            }
        return Response(result_obj)


# 제품 추천 API
class ProductRecommendView(APIView):
    def post(self, request):
        # 필수항목 체크
        serializer = RecommendInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({
                "status": 400,
                "message": "Invalid input",
                "errors": serializer.errors,
                "result": None
            }, status=400)

        # 점수가 제일 낮은 2개 항목에 대해 추천 제품 3개씩 조회
        scores = {
            'wrinkle': int(request.data.get('wrinkle')),
            'pore': int(request.data.get('pore')),
            'elasticity': int(request.data.get('elasticity')),
            'moisture': int(request.data.get('moisture')),
        }

        lowest = sorted(scores.items(), key=lambda x: x[1])[:2]
        lowest_features = [feature for feature, _ in lowest]

        recommended = []
        for feature in lowest_features:
            products = ProductMgmt.objects.select_related('company').order_by(F(feature).desc())[:3]
            recommended += list(products)

        # ID 기준 중복 제거
        unique_recommendations = {p.id: p for p in recommended}.values()

        serializer = ProductMgmtSerializer(unique_recommendations, many=True)

        return Response({
            "status": 200,
            "message": "ok",
            "result": serializer.data
        })
