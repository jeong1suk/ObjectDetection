import torch
import torch.nn as nn
import torchvision

class YOLOv1_RESNET(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.num_bboxes = 2
        self.grid_size = 7
        
        # ResNet18 백본 네트워크
        resnet18 = torchvision.models.resnet18(pretrained=True)
        layers = [m for m in resnet18.children()]
        self.backbone = nn.Sequential(*layers[:-2])
        
        # YOLO 헤드 네트워크
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            # 출력 레이어: (4+1)*num_bboxes + num_classes
            # 4: bbox 좌표(x,y,w,h), 1: objectness score
            nn.Conv2d(
                in_channels=1024, 
                out_channels=(4+1)*self.num_bboxes+self.num_classes, 
                kernel_size=1, 
                padding=0, 
                bias=False
            ),
            nn.AdaptiveAvgPool2d(output_size=(self.grid_size, self.grid_size))
        )
        
    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)
        return out

    @torch.no_grad()
    def predict(self, image, conf_thres=0.3, iou_threshold=0.1):
        """
        모델 추론을 위한 메서드
        Args:
            image: 입력 이미지 텐서 (배치, 채널, 높이, 너비)
            conf_thres: confidence threshold
            iou_threshold: NMS를 위한 IoU threshold
        Returns:
            bboxes: 감지된 박스 좌표 (x,y,w,h)
            scores: 각 박스의 confidence score
            class_ids: 각 박스의 클래스 ID
        """
        predictions = self(image)
        prediction = predictions.detach().cpu().squeeze(dim=0)
        
        grid_size = prediction.shape[-1]
        y_grid, x_grid = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size))
        stride_size = image.shape[-1]/grid_size

        # 예측값 추출 및 변환
        conf = prediction[[0,5], ...].reshape(1, -1)
        xc = (prediction[[1,6], ...] * image.shape[-1] + x_grid*stride_size).reshape(1,-1)
        yc = (prediction[[2,7], ...] * image.shape[-1] + y_grid*stride_size).reshape(1,-1)
        w = (prediction[[3,8], ...] * image.shape[-1]).reshape(1,-1)
        h = (prediction[[4,9], ...] * image.shape[-1]).reshape(1,-1)
        cls = torch.max(prediction[10:, ...].reshape(self.num_classes, -1), dim=0).indices.tile(1,2)
        
        # bbox 좌표 계산
        x_min = xc - w/2
        y_min = yc - h/2
        x_max = xc + w/2
        y_max = yc + h/2

        prediction_res = torch.cat([x_min, y_min, x_max, y_max, conf, cls], dim=0)
        prediction_res = prediction_res.transpose(0,1)
            
        # Confidence threshold 적용
        pred_res = prediction_res[prediction_res[:, 4] > conf_thres]
        
        # NMS 적용
        nms_index = torchvision.ops.nms(
            boxes=pred_res[:, 0:4], 
            scores=pred_res[:, 4], 
            iou_threshold=iou_threshold
        )
        pred_res_ = pred_res[nms_index].numpy()
        
        # 결과 포맷팅
        n_obj = pred_res_.shape[0]
        bboxes = pred_res_[:, 0:4]
        bboxes[:, 0:2] = (pred_res_[:, 0:2] + pred_res_[:, 2:4]) / 2
        bboxes[:, 2:4] = pred_res_[:, 2:4] - pred_res_[:, 0:2]
        scores = pred_res_[:, 4]
        class_ids = pred_res_[:, 5]
        
        return bboxes, scores, class_ids 