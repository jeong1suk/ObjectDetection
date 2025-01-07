import os
import cv2
import torch
from torchvision.ops import nms

BOX_COLOR = {'Bus':(200, 0, 0), 'Truck':(0, 0, 200)}

TEXT_COLOR = (255, 255, 255)

CLASS_NAME_TO_ID = {
    'Truck': 0,
    'Bus': 1
}

CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}

def postprocess(prediction, conf_thres=0.2, iou_thres=0.1):
    """
    모델 예측 결과를 후처리합니다.
    
    Args:
        prediction (dict): 모델의 예측 결과
        conf_thres (float): Confidence threshold
        iou_thres (float): NMS IoU threshold
    
    Returns:
        processed_pred (torch.Tensor): 후처리된 예측 결과
        shape: [num_boxes, 6] (x1, y1, x2, y2, confidence, class_id)
    """
    boxes = prediction['boxes']
    scores = prediction['scores']
    labels = prediction['labels']
    
    # Confidence threshold
    mask = scores > conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # NMS 적용
    keep_indices = nms(boxes, scores, iou_thres)
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]
    
    if len(boxes) == 0:
        return torch.zeros((0, 6))
    
    # 결과 통합 [x1, y1, x2, y2, confidence, class_id]
    processed_pred = torch.cat((boxes, scores.unsqueeze(1), labels.unsqueeze(1)), dim=1)
    return processed_pred

def XminYminXmaxYmax_to_XminYminWH(boxes):
    """
    [xmin, ymin, xmax, ymax] 형식을 [xmin, ymin, width, height] 형식으로 변환
    
    Args:
        boxes (np.ndarray): [N, 4] 형태의 박스 좌표
    
    Returns:
        converted_boxes (np.ndarray): 변환된 박스 좌표
    """
    converted_boxes = boxes.copy()
    converted_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    converted_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
    return converted_boxes

def save_model(model_state, model_name, save_dir="./trained_model"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model_state, os.path.join(save_dir, model_name))


def visualize_bbox(image, bbox, class_name, color=BOX_COLOR, thickness=2):
    x_center, y_center, w, h = bbox
    x_min = int(x_center - w/2)
    y_min = int(y_center - h/2)
    x_max = int(x_center + w/2)
    y_max = int(y_center + h/2)
    
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color[class_name], thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(image, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color[class_name], -1)
    cv2.putText(
        image,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=TEXT_COLOR, 
        lineType=cv2.LINE_AA,
    )
    return image


def visualize(image, bboxes, category_ids):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = CLASS_ID_TO_NAME[category_id]
        img = visualize_bbox(img, bbox, class_name)
    return img