# PDC_2022
2022 PNU Deep Learning Challenge / CSE Tech Week

## Leap Forward Tech Week!
* 주제 : 내년 PNU CSE Tech Week 때 진행했으면 하는 프로그램 아이디어 내기 

* 접수 기간 : 10/09 ~ 10/25 - 부원끼리 모인 사진 한 장과 아이디어 보고서를 10월 25일까지 접수,  CTG 별로  최대 하나의  아이디어 제출 가능 

* 보고서 내용에는  1) 프로그램의 이름   2) 프로그램의 취지/기획 의도  3) 운영 계획 (시간, 후보 장소 등을 포함)  4) 준비 계획 (사전 준비 작업 계획) 등이 필히 들어가야 합니다.

보고서는 PDF 형식으로 출력하여 제출
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
