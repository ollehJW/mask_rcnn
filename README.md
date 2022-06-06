## Mask RCNN

Detectron 2 패키지를 활용한 Mask RCNN 활용 코드

## 환경 설정 (window 환경에선 detectron2 window버전 다운 필요, colab 환경에서 구동가능)
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2==0.5 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
pip install opencv-python
pip install -U git+https://github.com/facebookresearch/fvcore.git
pip install labelme
conda install scikit-learn



### 사용법
1. Data 폴더안에 bmp와 json 파일을 전부 넣어줍니다. 다른 포맷일 경우, 코드 내 bmp를 다른 포맷으로 바꾸자. (Labelme 포맷)

2. anaconda에 python data_split.py 를 입력하면 Data 폴더안에 train, test 폴더가 만들어지고 각 폴더안에 데이터가 분할됩니다.

3. anaconda에 python make_coco.py 를 입력하면 Data폴더안에 train.json과 test.json이 만들어집니다. (Coco 데이터 포맷)

4. anaconda에 python train.py 를 입력하면 학습이 진행되며, output 폴더가 생성됩니다. 해당 폴더안의 metrics.json에 학습 성능 정보가 기입되며, 학습이 종료되면 model_final이라는 모델 파일과 loss_graph 이미지 그리고 마지막으로 test 데이터에 대한 예측이 output 폴더내 visualization 폴더안에 저장되게 됩니다.

