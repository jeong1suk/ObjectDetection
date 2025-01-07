import cv2
import torch
import numpy as np
from time import time
from models.faster_rcnn import load_model
from data.transforms import get_transforms
from utils.utils import visualize
from config.config import Config

class Inferencer:
    def __init__(self, model_path):
        self.cfg = Config()
        self.model = load_model(model_path)
        self.model.eval()
        self.transformer = get_transforms('val')
        
    @torch.no_grad()
    def predict(self, image):
        # 이미지 전처리
        tensor_image = self.transformer(image)
        tensor_image = tensor_image.to(self.cfg.DEVICE)
        
        # 예측
        prediction = self.model([tensor_image])
        return prediction[0]
    
    def run_video(self, video_path):
        vid = cv2.VideoCapture(video_path)
        
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
                
            # 추론 시작
            since = time()
            ori_h, ori_w = frame.shape[:2]
            image = cv2.resize(frame, (self.cfg.IMAGE_SIZE, self.cfg.IMAGE_SIZE))
            prediction = self.predict(image)
            
            # 후처리
            prediction = self.postprocess_prediction(prediction, ori_w, ori_h)
            
            # FPS 계산
            fps_text = f"{(time() - since)*1000:.0f}ms/image"
            
            # 시각화
            canvas = visualize(frame, prediction)
            cv2.putText(canvas, fps_text, (20, 40), 
                       cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            cv2.imshow('Detection', canvas)
            
            if cv2.waitKey(1) == 27:  # ESC
                break
                
        vid.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "./trained_model/model_30.pth"
    video_path = "./sample_video.mp4"
    
    inferencer = Inferencer(model_path)
    inferencer.run_video(video_path) 