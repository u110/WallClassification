"""
クラスインデックス->クラス名
のディクショナリをclasses.pklファイルとして保存するだけのスクリプト。
予測するときに名前を表示するために必要なので。
"""

from keras.preprocessing.image import ImageDataGenerator
from pickle import dump
import argparse


if __name__ == "__main__":
    batch_size = 32
    input_shape = (224, 224, 3)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_dir", default='resized_val')
    args = parser.parse_args()
    train_dir = args.train_dir
    train_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=input_shape[0:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
    )

    print(train_generator.class_indices)
    print(train_generator.directory)
    with open("classes.pkl", "wb") as f:
        dump({index: key for key, index in train_generator.class_indices.items()}, f)
