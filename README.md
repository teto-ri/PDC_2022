# PDC_2022
2022 PNU Deep Learning Challenge

# Track1
Deep Learning을 이용한 이미지 분류 (Classification)

# Track2
Deep Learning을 이용한 예술 작품 생성 (Art Generation)

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
