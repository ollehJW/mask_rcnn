### 7. Test 데이터셋에 대한 실제 detection 시각화
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
from detectron2.config import get_cfg

cfg = get_cfg()

## 변경 O
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 500
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5



### 2. Test 데이터 경로 설정
test_image_dir = "/home/jongwook95.lee/study/object_detection/mask_rcnn/pytorch_mask/data/test"

test_set = glob(test_image_dir + "/*.bmp")
os.makedirs('./output/visualization', exist_ok=True)

def visualization(cfg, test_set):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    predictor = DefaultPredictor(cfg)
    for file_name in test_set:
        im = cv2.imread(file_name)
        file_base_name = os.path.basename(file_name).split('.')[0] 
        outputs = predictor(
            im)
        v = Visualizer(im[:, :, ::-1],
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
        plt.imsave(os.path.join('./output/visualization', file_base_name + '.png'), img)

visualization(cfg, test_set)