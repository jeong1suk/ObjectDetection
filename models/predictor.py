import torch
from config.config import Config
from utils.box_ops import postprocess

class Predictor:
    """
    객체 탐지 모델의 추론을 수행하는 클래스
    """
    def __init__(self, model):
        """
        Args:
            model: 학습된 Faster R-CNN 모델
        """
        self.cfg = Config()
        self.model = model
        self.model.to(self.cfg.DEVICE)
        self.model.eval()
        
    def predict_single_image(self, image):
        """
        단일 이미지에 대한 객체 탐지를 수행합니다.
        
        Args:
            image (torch.Tensor): 입력 이미지 텐서 [C, H, W]
            
        Returns:
            boxes (torch.Tensor): 탐지된 바운딩 박스 좌표
            labels (torch.Tensor): 탐지된 객체의 클래스 레이블
            scores (torch.Tensor): 탐지 신뢰도 점수
        """
        self.model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.cfg.DEVICE)  # [1, C, H, W]
            predictions = self.model(image)[0]
            
            # 후처리 (NMS 등)
            boxes = predictions['boxes']
            scores = predictions['scores']
            labels = predictions['labels']
            
            # confidence threshold 적용
            keep_idxs = scores > self.cfg.CONF_THRESHOLD
            boxes = boxes[keep_idxs]
            scores = scores[keep_idxs]
            labels = labels[keep_idxs]
            
            return boxes.cpu(), labels.cpu(), scores.cpu()
    
    def predict_batch(self, images):
        """
        배치 이미지에 대한 객체 탐지를 수행합니다.
        
        Args:
            images (List[torch.Tensor]): 입력 이미지 텐서들의 리스트
            
        Returns:
            batch_boxes (List[torch.Tensor]): 각 이미지별 탐지된 바운딩 박스 좌표
            batch_labels (List[torch.Tensor]): 각 이미지별 탐지된 객체의 클래스 레이블
            batch_scores (List[torch.Tensor]): 각 이미지별 탐지 신뢰도 점수
        """
        self.model.eval()
        batch_boxes = []
        batch_labels = []
        batch_scores = []
        
        with torch.no_grad():
            predictions = self.model(images)
            
            for pred in predictions:
                boxes = pred['boxes']
                scores = pred['scores']
                labels = pred['labels']
                
                # confidence threshold 적용
                keep_idxs = scores > self.cfg.CONF_THRESHOLD
                boxes = boxes[keep_idxs]
                scores = scores[keep_idxs]
                labels = labels[keep_idxs]
                
                batch_boxes.append(boxes.cpu())
                batch_labels.append(labels.cpu())
                batch_scores.append(scores.cpu())
        
        return batch_boxes, batch_labels, batch_scores