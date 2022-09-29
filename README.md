# PDC_2022
2022 PNU Deep Learning Challenge

## Track1
* Deep Learning을 이용한 이미지 분류 (Classification)
부산대학교와 정컴을 대표하는 Landmark 5종류에 대한 영상 분류를 위한 Deep Learning
 * 랜드마크 영상 예시
![img](https://imgur.com/a/LtFsdVh)

* 참가 팀은 Training / Test Dataset을 직접 수집하여 사용 할 수 있다.
* Training & Testing Dataset, Trained Model 및 개발 내용 요약문을 포함하여 10/25 까지 접수
* 접수 기간 종료 후 각 참여 팀의 Test Dataset에서 일정 수량을 Random Sampling 하여 최종 성능 평가 실시
* Dataset 및 Model Submission Format (e.g., Python Class 명세 등)은 추후 제출 링크 공개 시 안내 예정

### Challege Details
* 입력 데이터의 형태는 Image 이며, Image Size는 1280 x 960 이하 이다.
* 제출된 코드는 입력 영상의 크기에 상관없이 동작해야 한다.

## Track2
* Deep Learning을 이용한 예술 작품 생성 (Art Generation)
* AI 기반의 Image Generation을 이용해 부산 / 부산대학교 / IT / 정컴을 대표 할 수 있는 이미지 형태의 예술 작품을 생성
* AI 기반의 이미지 생성 및 Retouching 과정등을 포함한 요약문을 10/25 까지 접수
* 작품은 AI를 이용해 생성하였음을 보여야 함
* 생성과정을 간단한 보고서로 만들어 함께 접수
* 이미지 형태의 작품 및 작품 설명을 포함하여 접수
* 접수 기간 종료 후 각 참여 팀의 작품을 Tech Week Website에 게시하고, 참가자들의 Online 투표를 통해 우승 작품 선발
* Image Generation, Style Transfer, Blending 등의 다양한 기법 활용 가능

## Dependency Management
* Sever : Window10 , Cuda 11.3, CuDNN: 8.2.1, Python 3.9.13

* Sever specification : AMD Ryzen 3 3300X, 16GB ram, GTX1660SUPER

## Project Management
1. /config : Project config as json, ini..
2. /data : Training data as train, val, test
3. /docs : PDC2022 document as Notice, report..
4. /checkpoints : Trained models with name rules
5. /notebooks : data processing, Modeling ipynb notebook
6. /scripts : Train,test,run py scripts
7. /test : Sample, Unit test
8. /src : Program sources

### /configs

- model configs
    + Choose the format
    + `.json`
    + `.ini`
    + `.yaml`
    + `.py`

### /data

- your datasets
    + train
        - class 1
        - class 2
        - etc.
    + valid
        - class 1
        - class 2
        - etc.
    + test
        - class 1
        - class 2
        - etc.

### /docs

- project documents
    + install
    + how to run?
    + api structure

### /checkpoints

- trained model

### /notebooks

- data processing notebook
- tutorial notebook

### /scripts

- train scripts
- test scripts
- run scripts

### /test

For unittest

- dataset
- dataloader
- model
- train
- valid
- test

### /src

- `/data`
    + dataloader
    + data utils

- `/model`
    + model architecture
    + model utils

- `/visualization`
    + visualization tools

- `loss.py` and `/loss`
    + define custom loss

- `optim.py` and `/optim`
    + define custom optimizer

- `scheduler.py` and `/scheduler`
    + define custom scheduler

- `transforms.py` and `/transforms`
    + define custom data augmentation

### setup.py
