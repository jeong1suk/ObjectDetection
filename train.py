import os
import torch
import logging
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from config.config import Config
from models.faster_rcnn import build_model
from data.dataloader import build_dataloader
from utils.metrics import evaluate_coco

class Trainer:
    def __init__(self):
        self.cfg = Config()
        self.device = self.cfg.DEVICE
        
        # 모델 생성
        self.model = build_model()
        self.model.to(self.device)
        
        # 옵티마이저 설정
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.cfg.LEARNING_RATE,
            momentum=self.cfg.MOMENTUM,
            weight_decay=self.cfg.WEIGHT_DECAY
        )
        
        # 데이터로더 생성
        self.dataloaders = {
            'train': build_dataloader(self.cfg.DATA_DIR, 'train'),
            'val': build_dataloader(self.cfg.DATA_DIR, 'val')
        }
        
        # 로깅 설정
        self.writer = self._init_tensorboard()
        self._init_logging()
        
    def _init_tensorboard(self):
        """Tensorboard 설정"""
        log_dir = os.path.join(
            'runs',
            datetime.now().strftime('%Y%m%d-%H%M%S')
        )
        return SummaryWriter(log_dir)
    
    def _init_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            filename=f'training_{datetime.now():%Y%m%d_%H%M%S}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def train_one_epoch(self, epoch):
        """한 에폭 학습"""
        self.model.train()
        epoch_loss = 0
        
        with tqdm(self.dataloaders['train'], desc=f'Epoch {epoch}') as pbar:
            for i, (images, targets, _) in enumerate(pbar):
                # 데이터 준비
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} 
                          for t in targets]
                
                # 순전파 및 손실 계산
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # 역전파 및 옵티마이저 스텝
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                
                # 손실 기록
                epoch_loss += losses.item()
                
                # 프로그레스바 업데이트
                pbar.set_postfix({'loss': losses.item()})
                
                # Tensorboard에 기록
                if i % self.cfg.VERBOSE_FREQ == 0:
                    step = epoch * len(self.dataloaders['train']) + i
                    self.writer.add_scalar('Loss/train', losses.item(), step)
                    
        return epoch_loss / len(self.dataloaders['train'])
    
    @torch.no_grad()
    def validate(self, epoch):
        """검증 수행"""
        self.model.eval()
        val_stats = evaluate_coco(
            self.model,
            self.dataloaders['val'],
            self.device
        )
        
        # Tensorboard에 메트릭 기록
        self.writer.add_scalar('AP@0.5/val', val_stats[1], epoch)  # mAP@0.5
        self.writer.add_scalar('AP@0.75/val', val_stats[2], epoch)  # mAP@0.75
        
        return val_stats[0]  # mAP@0.5:0.95 반환
    
    def save_checkpoint(self, epoch, val_map):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_map': val_map
        }
        
        save_path = f'checkpoints/checkpoint_epoch{epoch}_map{val_map:.3f}.pth'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, save_path)
        logging.info(f'Checkpoint saved: {save_path}')
    
    def train(self):
        """전체 학습 과정"""
        logging.info("=== Training Started ===")
        best_map = 0
        
        for epoch in range(self.cfg.NUM_EPOCHS):
            # 학습
            train_loss = self.train_one_epoch(epoch)
            logging.info(f'Epoch {epoch} - Train Loss: {train_loss:.4f}')
            
            # 검증
            val_map = self.validate(epoch)
            logging.info(f'Epoch {epoch} - Validation mAP: {val_map:.4f}')
            
            # 체크포인트 저장
            if val_map > best_map:
                best_map = val_map
                self.save_checkpoint(epoch, val_map)
        
        logging.info("=== Training Finished ===")
        self.writer.close()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train() 