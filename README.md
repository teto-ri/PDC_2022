# PDC_2022
2022 PNU Deep Learning Challenge

## Track1
* Deep Learning을 이용한 이미지 분류 (Classification)
https://github.com/PNUCSE/2022_DLC_LandmarkClassification
https://docs.google.com/document/u/1/d/1PVFJimsrsI_e9ye-1NfXhdgDlyimA--yBl92OGhi7zU/edit#heading=h.76moldfr47w0

* 부산대학교와 정컴을 대표하는 Landmark 5종류에 대한 영상 분류를 위한 Deep Learning

* 참가 팀은 Training / Test Dataset을 직접 수집하여 사용 할 수 있다.
* Training & Testing Dataset, Trained Model 및 개발 내용 요약문을 포함하여 10/25 까지 접수
* 접수 기간 종료 후 각 참여 팀의 Test Dataset에서 일정 수량을 Random Sampling 하여 최종 성능 평가 실시
* Dataset 및 Model Submission Format (e.g., Python Class 명세 등)은 추후 제출 링크 공개 시 안내 예정

### Challege Details
* 입력 데이터의 형태는 Image 이며, Image Size는 1280 x 960 이하 이다.
* 제출된 코드는 입력 영상의 크기에 상관없이 동작해야 한다.

## Track2
* Deep Learning을 이용한 예술 작품 생성 (Art Generation)
https://docs.google.com/document/d/1SozR4Z0wgjDETKGn1pL6tL54vaDsehVZOZ_-pacU_6o/edit

* AI 기반의 Image Generation을 이용해 부산 / 부산대학교 / IT / 정컴을 대표 할 수 있는 이미지 형태의 예술 작품을 생성
* AI 기반의 이미지 생성 및 Retouching 과정등을 포함한 요약문을 10/25 까지 접수
* 작품은 AI를 이용해 생성하였음을 보여야 함
* 생성과정을 간단한 보고서로 만들어 함께 접수
* 이미지 형태의 작품 및 작품 설명을 포함하여 접수
* 접수 기간 종료 후 각 참여 팀의 작품을 Tech Week Website에 게시하고, 참가자들의 Online 투표를 통해 우승 작품 선발
* Image Generation, Style Transfer, Blending 등의 다양한 기법 활용 가능

## Dependency Management
* Sever : Window10 , Cuda 11.3, CuDNN: 8.2.1, Python 3.9.13
* Main Framework : Tensorflow 2.8, OpenCV 4.6
* Sever specification : AMD Ryzen 3 3300X, 16GB ram, GTX3060


## Dataset Structure
- Train 과 Test 데이터셋 구조는 동일합니다.
- Root directory (train 또는 test) 아래에는 5개 Landmark에 대한 subdir이 존재합니다.
- 각 클래스는 **cse(0, 컴퓨터공학관), hh(1, Humanities Hall, 인문관), rg(2, 무지개문), wb(3, 웅비의 탑), wjj(4, 운죽정)** 에 해당합니다.
- 예제 데이터의 경우 7장의 train, 3장의 test image가 각 class 별로 포함되어 있으며 png format 입니다.
- 다만 실제 데이터의 경우 png, jpg, bmp 등의 다양한 이미지 포맷이 포함되어 있을 수 있습니다.
- 예제 데이터의 경우 이름의 형식을 통일해 두었으나, 실제 데이터의 경우 임의의 파일 이름을 가질 수 있습니다.
- 이미지의 최소 사이즈는 가로 x 세로 기준 320 x 240, 최대 사이즈는 1280 x 960 로 제한합니다.

```commandline
dataset
├── test
│   ├── cse
│   │   ├── 00000000.png
│   │   ├── ...
│   │   └── XXXXXXXX.png
│   ├── hh
│   │   ├── 00000000.png
│   │   ├── ...
│   │   └── XXXXXXXX.png
│   ├── rg
│   │   ├── 00000000.png
│   │   ├── ...
│   │   └── XXXXXXXX.png
│   ├── wb
│   │   ├── 00000000.png
│   │   ├── ...
│   │   └── XXXXXXXX.png
│   └── wjj
│       ├── 00000000.png
│       ├── ...
│       └── XXXXXXXX.png
└── train
    ├── cse
    │   ├── 00000000.png
    │   ├── ...
    │   └── XXXXXXXX.png
    ├── hh
    │   ├── 00000000.png
    │   ├── ...
    │   └── XXXXXXXX.png
    ├── rg
    │   ├── 00000000.png
    │   ├── ...
    │   └── XXXXXXXX.png
    ├── wb
    │   ├── 00000000.png
    │   ├── ...
    │   └── XXXXXXXX.png
    └── wjj
        ├── 00000000.png
        ├── ...
        └── XXXXXXXX.png
```
