
"""
필수 라이브러리:
pip install torch torchvision Pillow numpy

필요 파일:
- skin_model.pth

입력 형태:
- base64 인코딩된 이미지 문자열만 허용
- data URL 형태: "data:image/jpeg;base64,/9j/4AAQ..."
- 순수 base64: "/9j/4AAQSkZJRgABAQAAAQ..."
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from torchvision import models
import base64
import io, os
from django.conf import settings

# 상수 정의
FACEPART_MAP = {
    1: 'forehead',     # 이마
    3: 'l_perocular',  # 왼쪽 눈가
    4: 'r_perocular',  # 오른쪽 눈가
    5: 'l_cheek',      # 왼쪽 볼
    6: 'r_cheek',      # 오른쪽 볼
    8: 'chin'          # 턱
}

# 파라미터 맵
PART_PARAMETERS = {
    'forehead': ['moisture', 'elasticity'],
    'l_cheek': ['moisture', 'elasticity', 'pore'],
    'r_cheek': ['moisture', 'elasticity', 'pore'],
    'l_perocular': ['wrinkle'],
    'r_perocular': ['wrinkle'],
    'chin': ['moisture', 'elasticity']
}

# 정규화 범위
NORMALIZATION_RANGES = {
    'moisture': {
        'forehead': (26.0, 91.667),
        'l_cheek': (27.5, 79.333),
        'r_cheek': (27.5, 79.333),
        'chin': (26.667, 91.0)
    },
    'elasticity': {  # Q0 지표만 사용
        'forehead': (20.0, 85.0),
        'l_cheek': (25.0, 85.0),
        'r_cheek': (25.0, 85.0),
        'chin': (15.0, 75.0)
    },
    'wrinkle': {  # Ra 지표만 사용
        'l_perocular': (9.9877, 49.13),
        'r_perocular': (10.415, 45.682)
    },
    'pore': {
        'l_cheek': (235.0, 1840.0),
        'r_cheek': (235.0, 1840.0)
    }
}

class SkinAnalysisModel(nn.Module):
    """Precision Edition 모델 구조"""
    def __init__(self, num_parts=6, max_params=3):
        super(SkinAnalysisModel, self).__init__()

        # MobileNetV2 백본 (ImageNet 사전훈련)
        mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.backbone = mobilenet.features

        # 특성 추출기 (BatchNorm 추가)
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(1280 * 49, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # 부위 분류 헤드 (정면 전용)
        self.part_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_parts)
        )

        # 파라미터 예측 헤드 (Q0/Ra 지표 전용, 부위별)
        self.parameter_heads = nn.ModuleDict({
            'forehead': nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2),       # moisture, elasticity_Q0
                nn.Sigmoid()
            ),
            'l_cheek': nn.Sequential(
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3),       # moisture, elasticity_Q0, pore
                nn.Sigmoid()
            ),
            'r_cheek': nn.Sequential(
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3),       # moisture, elasticity_Q0, pore
                nn.Sigmoid()
            ),
            'l_perocular': nn.Sequential(
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),       # wrinkle_Ra
                nn.Sigmoid()
            ),
            'r_perocular': nn.Sequential(
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),       # wrinkle_Ra
                nn.Sigmoid()
            ),
            'chin': nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2),       # moisture, elasticity_Q0
                nn.Sigmoid()
            )
        })

        self.part_names = list(FACEPART_MAP.values())
        self.max_params = max_params

    def forward(self, x):
        # 백본 특성 추출
        features = self.backbone(x)
        features = self.feature_extractor(features)

        # 부위 분류
        part_pred = self.part_classifier(features)

        # 파라미터 예측 (모든 부위에 대해)
        param_preds = []
        for part_name in self.part_names:
            pred = self.parameter_heads[part_name](features)
            param_preds.append(pred)

        return {
            'part': part_pred,
            'parameters': param_preds
        }

class SkinAnalyzer:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(settings.BASE_DIR, 'skinni_main', 'skin_model.pth')
        print(f"model_path:{model_path}")
        self.device = torch.device('cpu')
        
        # 모델 로드
        self.model = SkinAnalysisModel()
        
        try:
            # 가중치만 로드
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        except:
            # 전체 체크포인트 로드
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        # 이미지 전처리 (ImageNet 정규화)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def denormalize_parameter(self, part_name, param_name, normalized_value):
        # 정규화 값을 실제 값으로 복원
        if param_name not in NORMALIZATION_RANGES:
            return normalized_value
        
        if part_name not in NORMALIZATION_RANGES[param_name]:
            return normalized_value
        
        min_val, max_val = NORMALIZATION_RANGES[param_name][part_name]
        return normalized_value * (max_val - min_val) + min_val

    def analyze_image(self, base64_image):
        """
        base64 이미지 분석
        
        Args:
            base64_image (str): base64 인코딩된 이미지 문자열
        
        Returns:
            dict: 6개 부위별 파라미터 + 4개 파라미터 평균값
        """
        try:
            # base64 처리
            if base64_image.startswith('data:image'):
                base64_data = base64_image.split(',')[1]
            else:
                base64_data = base64_image
            
            # PIL Image 변환
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # 전처리
            tensor = self.transform(image).unsqueeze(0)
            
            # 모델 추론
            with torch.no_grad():
                outputs = self.model(tensor)
                param_preds = outputs['parameters']
            
            # 부위별 파라미터 분석
            parts_analysis = {}
            
            for part_idx, part_name in enumerate(self.model.part_names):
                param_names = PART_PARAMETERS[part_name]
                pred_values = param_preds[part_idx][0].cpu().numpy()
                
                part_data = {}
                for i, param_name in enumerate(param_names):
                    if i < len(pred_values):
                        normalized_value = float(pred_values[i])
                        # 실제 값으로 복원
                        actual_value = self.denormalize_parameter(part_name, param_name, normalized_value)
                        part_data[param_name] = actual_value
                
                parts_analysis[part_name] = part_data
            
            # 4개 파라미터 평균 계산 (실제 값 기준)
            param_totals = {'moisture': [], 'elasticity': [], 'wrinkle': [], 'pore': []}
            
            for part_name, part_data in parts_analysis.items():
                for param_name, value in part_data.items():
                    if param_name in param_totals:
                        param_totals[param_name].append(value)
            
            averages = {}
            for param_name, values in param_totals.items():
                if values:
                    if param_name == 'pore':
                        # pore는 값이 낮을수록 좋음 (모공 수니까)
                        avg_value = sum(values) / len(values)
                        # 정규화 범위 기준으로 백분율 계산 후 역전
                        pore_min, pore_max = 235.0, 1840.0
                        normalized_avg = (avg_value - pore_min) / (pore_max - pore_min)
                        averages[param_name] = 100 - (normalized_avg * 100)
                    elif param_name == 'wrinkle':
                        # wrinkle도 값이 낮을수록 좋음 (주름이니까)
                        avg_value = sum(values) / len(values)
                        # 정규화 범위 기준으로 백분율 계산 후 역전
                        # l_perocular, r_perocular 만 해당
                        wrinkle_min = (9.9877 + 10.415) / 2
                        wrinkle_max = (49.13 + 45.682) / 2
                        normalized_avg = (avg_value - wrinkle_min) / (wrinkle_max - wrinkle_min)
                        averages[param_name] = 100 - (normalized_avg * 100)
                    else:
                        # moisture, elasticity는 정상적으로 (높을수록 좋음)
                        averages[param_name] = sum(values) / len(values)
                else:
                    averages[param_name] = 0.0

            return {
                'success': True,
                'parts': parts_analysis,
                'averages': averages,
                'model_version': '최종'
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

# 싱글톤
_analyzer = None

def analyze_skin_image(base64_image):
    """
    피부 분석 API 함수

    Args:
        base64_image (str): base64 인코딩된 이미지 문자열

    Returns:
        dict: 6개 부위별 파라미터 + 4개 파라미터 평균값 (Q0/Ra 지표 기반)
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = SkinAnalyzer()
    return _analyzer.analyze_image(base64_image)

def get_model_info():
    return {
        'model_type': 'MobileNetV2',
        'model_version': '최종',
        'total_parts': 6,
        'parameter_types': ['moisture', 'elasticity (Q0)', 'wrinkle (Ra)', 'pore'],
        'specialization': 'Front-only model with Q0/Ra indicators'
    }