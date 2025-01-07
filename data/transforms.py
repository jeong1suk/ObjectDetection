import torch
import torchvision.transforms as T
from config.config import Config

def get_transforms(phase: str):
    """
    학습/검증에 사용될 이미지 변환 함수를 반환합니다.
    
    Args:
        phase (str): 'train' 또는 'val'
    
    Returns:
        transforms (transforms.Compose): 변환 함수들의 composition
    """
    cfg = Config()
    
    if phase == 'train':
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
    else:
        transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
    
    return transforms
