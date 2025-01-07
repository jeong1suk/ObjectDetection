class Config:
    # 데이터 관련
    DATA_DIR = "../DATASET/Detection/"
    IMAGE_SIZE = 448
    NUM_CLASSES = 2
    
    # 학습 관련
    BATCH_SIZE = 6
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    MOMENTUM = 0.9
    
    # 추론 관련
    CONF_THRESHOLD = 0.2
    IOU_THRESHOLD = 0.1
    
    # 기타
    DEVICE = "cuda"
    VERBOSE_FREQ = 200
