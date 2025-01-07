import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from config.config import Config

def build_model():
    """
    Faster R-CNN 모델을 생성하고 설정합니다.
    
    Returns:
        model: 설정된 Faster R-CNN 모델
    """
    cfg = Config()
    
    # 사전 학습된 모델 로드
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # 백본 네트워크 고정 (선택사항)
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # anchor 설정
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    model.rpn.anchor_generator = anchor_generator
    
    # ROI pooler 설정
    model.roi_heads.box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )
    
    # 분류기 헤드 수정
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        cfg.NUM_CLASSES  # 배경 클래스 포함
    )
    
    return model

def load_model(model_path):
    """
    저장된 모델을 로드합니다.
    
    Args:
        model_path (str): 모델 가중치 파일 경로
        
    Returns:
        model: 로드된 모델
    """
    cfg = Config()
    model = build_model()
    model.load_state_dict(torch.load(model_path))
    model.to(cfg.DEVICE)
    return model