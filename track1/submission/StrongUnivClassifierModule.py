"""
    PNU CSE TECH WEEK 2022

    PNU Deep Learning Challenge - Track 01. Landmark Classification

    This script contains classifier specification for challenge submission.
    It is recommended to inherit from this class to implement your classifier.

    Jinsun Park (jspark@pusan.ac.kr / viplab@pusan.ac.kr)
    Visual Intelligence and Perception Lab., CSE, PNU

    ======================================================================

    2022.10.09 - Initial release

"""

import tensorflow as tf

class StrongUnivClassifier():
    """
    DLC-T1 Submission을 위한 Class 명세
    """
    
    def __init__(self, path_data, pretrain=None):
        """
        1. 생성자는 아래와 같은 인자만 사용해야 합니다.
        :param path_data: Dataset root에 해당하는 경로
        :param pretrain: Evaluation에 필요한 Pretrained weight의 경로

        2. 데이터 처리를 위한 transform과 dataset을 반드시 생성자에서 초기화 해야 합니다.
        3. 모델은 반드시 self.build_model method 내부에서 선언해 사용합니다.
        4. forward method는 입력 이미지 x를 받아서 (class score) y를 반환합니다.
        y는 normalized(sum(y) == 1) / unnormalized(sum(y) != 1) 여부에 상관 없이,
        argmax(y)의 값이 class index를 반환 할 수 있으면 됩니다.
        5. train_model은 반드시 존재해야 하지만, overriding 가능합니다.
        6. eval_model은 반드시 존재 할 필요는 없습니다. 참고용으로 사용하세요.
        7. Official evaluation은 self.model과 self.dataset을 직접 접근하여
        진행합니다. 절대 두 변수 이름을 바꾸지 마세요.
        8. self.dataset을 ImageFolder 이외의 class로 구현 할 경우, 각 샘플은
        반드시 (image, label)을 반환해야 합니다. (train / eval 코드 참조)

        특별한 이유가 없다면, ExampleClassifier를 상속하여 본인의 알고리즘을
        구현 한 뒤 제출하기를 추천합니다.
        각 method에 대한 추가적인 설명은 각 method의 docstring을 참고 해 주세요.
        """
        super().__init__()

        # Please refer to dataset directory structure
        self.path_data = path_data

        if pretrain is not None:
            # For evaluation
            self.model = tf.keras.models.load_model(pretrain)
        else:
            self.build_model()

        # Dataset loading에 적용하기 위한 transform은 생성자에서 선언
        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)

        # 데이터셋 구조가 정해져 있으므로, ImageFolder class를 사용하기를 추천
        # 다른 class를 사용 할 경우 반드시 각 샘플은 (image, label)을 반환해야 함.
        self.dataset = test_generator.flow_from_directory(
        directory=self.path_data,
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=1,
        shuffle=False
        )
        self.num_data = int(self.dataset.samples)

    def build_model(self):
        """
        Code 점검의 편의를 위해 model 선언은 반드시 build_model 안에서 완료해야 합니다.
        build_model 외부에서 model을 변경하지 마세요. (제발 Plz ^^)
        """
        kwargs = {'input_shape':(224, 224, 3),
                'include_top':False,
                'weights':'imagenet',
                'pooling':'avg'}
        pretrained_model = tf.keras.applications.ResNet50V2(**kwargs)
        pretrained_model.trainable = False
        
        tuning_layer_name = 'conv5_block1_preact_bn'
        tuning_layer = pretrained_model.get_layer(tuning_layer_name)
        tuning_index = pretrained_model.layers.index(tuning_layer)

        for layer in pretrained_model.layers[tuning_index:]:
            layer.trainable = True
        
        inputs = pretrained_model.input

        x = tf.keras.layers.Flatten()(pretrained_model.output)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
        return model
    
    def forward(self, x):
        """
        forward method는 입력 이미지 x를 받아서 (class score) y를 반환합니다.
        y는 normalized(sum(y) == 1) / unnormalized(sum(y) != 1) 여부에 상관 없이,
        argmax(y)의 값이 class index를 반환 할 수 있으면 됩니다.

        입력과 출력 부분 외에는 자유롭게 변형하여 사용하세요.

        :param x: 입력 영상 (Batch x Channel x Height x Width)
        :return: Class score (unnormalized or normalized)
        """
        # x: [Batch, Channel, Height, Width]
        # y: [Batch x Num_Class(5)]
        y = self.model.predict(x)
        return y

    def train_model(self, model, config):
        """
        train_model은 반드시 존재해야 하지만, overriding 가능합니다.
        기본으로 제공되는 코드는 구현 참고용 입니다.
        config의 내용은 train.py를 참고 하세요.

        train_model의 호출 이후에는 self.model의 weight가 훈련 완료 된 상태로 간주합니다.

        :param config: dictionary containing all of the training parameters
        :return:
        """
        batch_size = 32
        epochs = 6
        loss = 0.001
        optim = 'adam'
        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input,
        validation_split=0.1)
        
        train_images = train_generator.flow_from_directory(
        directory=self.path_data,
        subset='training',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed =0
        )
        val_images = train_generator.flow_from_directory(
        directory=self.path_data,
        subset='validation',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed = 0
        )
        print('Number of data : {}'.format(train_images.samples + val_images.samples))

        # Train
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint/epoch-{epoch}-val_loss-{val_loss:.4f}-val_acc-{val_accuracy:.4f}.h5', monitor='val_loss', save_best_only=True, verbose=1)
        history = model.fit(train_images,validation_data=val_images,validation_steps=5,epochs=epochs,verbose=1,callbacks=[model_checkpoint])
        # Get the best saved weights
        return history
        # 가장 기본적인 훈련 코드 구현의 예시
        