"""
サンプルのモデルの定義と訓練を行うモジュール
vgg16をベースにした転移学習モデルとなっている。
"""
from keras.models import Model,Sequential
import tensorflow as tf
from keras.layers.convolutional import Conv2D
from keras.layers import Activation, Dense, GlobalAveragePooling2D, Input, InputLayer, Lambda, Dropout, BatchNormalization
from keras.backend import sigmoid

import keras.applications
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard, EarlyStopping
import sys
import numpy as np
from collections import Counter
import continue_fit as cf
from keras.utils import multi_gpu_model
from keras import regularizers


class ModelFactory:

    MODELS = {
            "vgg16": keras.applications.vgg16.VGG16,
            "vgg19": keras.applications.vgg19.VGG19,
            "resnet50": keras.applications.resnet50.ResNet50,
            "inception_v3": keras.applications.inception_v3.InceptionV3,
            "mobilenet": keras.applications.mobilenet.MobileNet,
            "densenet": keras.applications.densenet.DenseNet201,
            "xception": keras.applications.xception.Xception,
            "nasnet": keras.applications.nasnet.NASNetLarge,
    }
    PREPROCESS_INPUT = {
            "vgg16": keras.applications.vgg16.preprocess_input,
            "vgg19": keras.applications.vgg19.preprocess_input,
            "resnet50": keras.applications.resnet50.preprocess_input,
            "inception_v3": keras.applications.inception_v3.preprocess_input,
            "mobilenet": keras.applications.mobilenet.preprocess_input,
            "densenet": keras.applications.densenet.preprocess_input,
            "xception": keras.applications.xception.preprocess_input,
            "nasnet": keras.applications.nasnet.preprocess_input,
    }

    @classmethod
    def get_model_cls(cls, model_name):
        return cls.MODELS.get(model_name)

    @classmethod
    def get_preprocess_input(cls, model_name):
        return cls.PREPROCESS_INPUT.get(model_name)


def get_based_model(input_shape, n_categories, fulltraining=False, base_model_cls=VGG16):
    """
    VGG16をベースにした転移学習モデルを生成する。
    fulltraining: Trueにすると、ベースモデルも含めて訓練可能にする。訓練速度が非常に遅くなる。
    """
    base_model = base_model_cls(
            weights='imagenet',
            include_top=False,
            input_tensor=Input(shape=input_shape)
    )

    #add new layers instead of FC networks
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Dropout(0.5)(x)
    x=Dense(1024)(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Dropout(0.5)(x)
    prediction=Dense(n_categories,activation='softmax')(x)
    model=Model(inputs=base_model.input,outputs=prediction)

    if not fulltraining:
        # fix weights
        for layer in base_model.layers:
            layer.trainable = False

    return model


import argparse


if __name__ == "__main__":
    # コマンドライン引数の定義/評価
    batch_size=32
    input_shape = (224,224,3)

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="保存するモデルファイルの名前、兼tensorBoardのログディレクトリ名")
    parser.add_argument("-t", "--train_dir", default='resized_cleaned', help="トレーニングデータセットが入っているディレクトリ")
    parser.add_argument("-v","--validation_dir",default='resized_val', help ="バリデーションデータセットが入っているディレクトリ")
    parser.add_argument("-b","--base_model",default='vgg16', help ="転移学習モデル")

    args = parser.parse_args()
    file_name = args.model_name
    train_dir=args.train_dir
    validation_dir=args.validation_dir
    base_model_name=args.base_model

    #訓練データの読み込み及びデータ拡張を行うための画像ジェネレータを生成。
    #VGG16用の前処理及び平行移動、回転、左右反転、シアー変換をランダムにかける。
    preprocess_input = ModelFactory.get_preprocess_input(base_model_name)
    train_datagen=ImageDataGenerator(
        preprocessing_function=preprocess_input, #VGG16の前処理
        height_shift_range=0.02,
        width_shift_range=0.02,
        shear_range=0.05,
        zoom_range=0.05,
        rotation_range=5,
        horizontal_flip=True,
        )
    train_generator=train_datagen.flow_from_directory(
        train_dir,
        target_size=input_shape[0:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
        )


    n_categories=len(train_generator.class_indices)
    #サンプルの多いクラスに予測が集中しないように、少ないサンプルのクラスほど重くなるように重みづけ
    class_weight ={ clss: len(train_generator.classes) / len(train_generator.class_indices) / count
                     for (clss,count) in Counter(train_generator.classes).most_common() }
    print("classes:",train_generator.class_indices)
    print("class weight:",class_weight)

    #バリデーションデータの画像読み込み処理を行うジェネレータを生成
    validation_datagen=ImageDataGenerator(
        preprocessing_function=preprocess_input,
        )
    validation_generator=validation_datagen.flow_from_directory(
        validation_dir,
        target_size=input_shape[0:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    base_model_cls = ModelFactory.get_model_cls(base_model_name)
    print("base_model_cls: %s" %  base_model_cls)
    model = get_based_model(
            input_shape,
            n_categories, 
            base_model_cls=base_model_cls)
    # parallel_model = multi_gpu_model(model, gpus=2)   #マルチGPUを使うときはこちら

    model.compile(optimizer=Adam(lr=1e-3),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    #訓練(中断しても続きから継続できる)
    hist=model.fit_generator(
        train_generator,
        epochs=30,
        initial_epoch=cf.load_epoch_init(file_name),
        # use_multiprocessing=True,
        verbose=1,
        workers=8,
        validation_data=validation_generator,
        class_weight=class_weight,
        callbacks=[
            CSVLogger(file_name+'.csv'),
            TensorBoard(file_name),
            cf.early_stopping(model, file_name),
            ])

