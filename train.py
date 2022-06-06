### 1. 사용할 패키지 불러오기
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import logging

### 2. 데이터 등록하기
"""
train_coco_dir: train annotation 정보를 coco로 변환한 json 파일 경로
train_image_dir: train image들이 저장된 경로
test_coco_dir: test annotation 정보를 coco로 변환한 json 파일 경로
test_image_dir: test image들이 저장된 경로

"""

train_coco_dir = "./data/train.json"
train_image_dir = "./data/train"
test_coco_dir = "./data/test.json"
test_image_dir = "./data/test"

register_coco_instances("train", {}, train_coco_dir, train_image_dir)
register_coco_instances("test", {}, test_coco_dir, test_image_dir)

### 3. 모델 파라미터 등록하기
"""
변경 O 파라미터만 조절해주는 것이 좋습니다.
BASE_LR : Learning Rate
MAX_ITER : Training Iteration (Epoch)
IMS_PER_BATCH: Batch size
NUM_CLASSES: Class 개수 (5개) <- 새로운 클래스가 추가되지 않는 한 그대로 두세요!

"""
## 변경 X
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.TEST.EVAL_PERIOD = 20

## 변경 O
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 500
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
mode = 'test'

### 4. Training 함수 정의
class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks

### 5. Training


if mode == 'train':
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume = False)
    trainer.train()


### 6. Loss 그래프 그리기.
def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines
import json

if mode == 'train':
    experiment_metrics = load_json_arr("./output/metrics.json")
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [15, 8]
    plt.plot(
        [x['iteration'] for x in experiment_metrics[:-1]], 
        [x['total_loss'] for x in experiment_metrics[:-1]])
    plt.plot(
        [x['iteration'] for x in experiment_metrics[:-1] if 'validation_loss' in x], 
        [x['validation_loss'] for x in experiment_metrics[:-1] if 'validation_loss' in x])
    plt.legend(['total_loss', 'validation_loss'], loc='upper left')
    plt.savefig('./output/loss_graph.png')

### 7. Test 데이터셋에 대한 실제 detection 시각화
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob


detecting_image_dir = "/home/jongwook95.lee/study/object_detection/mask_rcnn/pytorch_mask/data/detect"
test_set = glob(detecting_image_dir + "/*.bmp")
os.makedirs('./output/visualization', exist_ok=True)

class Metadata:
    def get(self, _):
        return ['etc','hole','line','scratch', 'short'] #your class labels


def visualization(cfg, test_set):
    import pandas as pd
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    predictor = DefaultPredictor(cfg)
    for file_name in test_set:
        im = cv2.imread(file_name)
        file_base_name = os.path.basename(file_name).split('.')[0] 
        outputs = predictor(
            im)
        x_left = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()[:, 0]
        x_right = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()[:, 2]
        y_up = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()[:, 1]
        y_down = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()[:, 3]
        bbox_class = outputs["instances"].to("cpu").pred_classes.tolist()
        bbox_info = pd.DataFrame({'x_left': x_left, 'x_right': x_right, 'y_up': y_up, 'y_down': y_down, 'class': bbox_class})
        bbox_info.to_csv(os.path.join('./output/visualization', file_base_name + '.csv'), index = False)
        
        
        v = Visualizer(im[:, :, ::-1],Metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
        plt.imsave(os.path.join('./output/visualization', file_base_name + '.png'), img)

visualization(cfg, test_set)