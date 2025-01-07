from torch.utils.data import DataLoader
from config.config import Config
from data.dataset import Detection_dataset
from data.transforms import get_transforms

def collate_fn(batch):
    """
    배치 데이터를 처리하기 위한 collate 함수
    
    Args:
        batch: DataLoader가 생성한 배치 데이터
        
    Returns:
        images (list): 이미지 텐서들의 리스트
        targets (list): 타겟 딕셔너리들의 리스트
    """
    image_list = []
    target_list = []
    filename_list = []
    
    for a,b,c in batch:
        image_list.append(a)
        target_list.append(b)
        filename_list.append(c)

    return image_list, target_list, filename_list
    

def build_dataloader(data_dir: str, phase: str):
    """
    데이터로더를 생성합니다.
    
    Args:
        data_dir (str): 데이터셋 경로
        phase (str): 'train' 또는 'val'
        
    Returns:
        dataloader (DataLoader): PyTorch DataLoader 객체
    """
    cfg = Config()
    
    # 데이터셋 생성
    dataset = Detection_dataset(
        data_dir=data_dir,
        phase=phase,
        transformer=get_transforms(phase)
    )
    
    # DataLoader 설정
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE if phase == 'train' else 1,
        shuffle=True if phase == 'train' else False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True if phase == 'train' else False
    )
    
    return dataloader
