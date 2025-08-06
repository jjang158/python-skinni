from rest_framework.views import APIView
from rest_framework.response import Response


class PickerTest(APIView):
    def post(self, request):
        file = request.data.get('file')
        file_info = request.data.get('file_info', '')
        
        print(f"Received file: {file}")
        print(f"File information : {file_info}")
        if not file or file_info == '':
            return Response({
                "status": 400,
                "message": "File or file information is missing",
                "result": {}
            })
            
        # 파일 설명
        data = {
            "file_description": "This is a test file for picker to text functionality.",
        }
        
        return Response({
            "status": 200,
            "message": "Success",
            "data": data
        })