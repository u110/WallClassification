"""
半教師在り学習を行うために、既存モデルを使って未クレンジング画像から確度の高い画像だけを抜き出すスクリプト
"""
from PIL import Image
from keras.models import load_model
import argparse
from erase_duplicates import walk
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import shutil
from keras.preprocessing.image import ImageDataGenerator
from pickle import dump, load


# Image.openが可能なパスだけフィルタリング
def openable_imgpaths(paths):
    for path in paths:
        try:
            Image.open(path)
            yield path
        except Exception as e:
            print("failed! open img file:", path, e)


class ImgGenerator():
    def __init__(self, paths: list, batch_size: int, target_size):
        self.paths = paths
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __iter__(self):
        for batch_paths in [
                self.paths[
                    i*self.batch_size: min((i+1)*self.batch_size, len(self.paths))
                ] for i in range(len(self))]:
            batch_img = np.array([img_to_array(load_img(path, target_size=self.target_size)) for path in batch_paths])
            yield preprocess_input(batch_img)

    def __getitem__(self, i: int):
        batch_paths = self.paths[i*self.batch_size: min((i+1)*self.batch_size, len(self.paths))]
        batch_img = np.array([img_to_array(load_img(path, target_size=self.target_size)) for path in batch_paths])
        return preprocess_input(batch_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="ラベル無し画像があるディレクトリ")
    parser.add_argument("output", help="出力先のディレクトリ")
    parser.add_argument("-m", "--model", default="vgg16_wallclassification_fine_cleaned_fulltrainingbest.h5")
    parser.add_argument("-b", "--batch_size", default=10, type=int)
    parser.add_argument("-t", "--threshold", default=0.9, type=float)
    parser.add_argument("-c", "--use_cache", help="キャッシュを利用するか否か", action="store_true")
    args = parser.parse_args()
    imgGen = ImageDataGenerator().flow_from_directory(args.input, batch_size=10)
    labels = {index: key for key, index in imgGen.class_indices.items()}
    imgPaths = list(openable_imgpaths(
        filter(lambda x: os.path.splitext(x)[1].lower() in [".jpg", ".png", ".jpeg"], walk(args.input))
    ))

    # print("len(gen) = ",len(gen), "len(imgPaths)=",len(imgPaths))
    pklname = "predicts.pkl"
    if os.path.exists(pklname) and args.use_cache:
        with open(pklname, "rb") as f:
            predicts = load(f)
    else:
        model = load_model(args.model)
        gen = ImgGenerator(imgPaths, args.batch_size, model.input_shape[1:-1])
        predicts = model.predict_generator(iter(gen), steps=len(gen), verbose=1, workers=8, max_queue_size=32)

    predict_labels = np.argmax(predicts, 1)
    ziped = zip(imgPaths, predicts)
    ziped = filter(lambda x: np.max(x[1]) > args.threshold, ziped)

    for path, predict in ziped:
        target = os.path.join(args.output, os.path.relpath(path, args.input))
        label = np.argmax(predict)
        label_name = labels[label]
        directory = os.path.dirname(target)

        if not directory.endswith(label_name):
            print("mismatch predicts and dirname! predict:", label_name, "filepath:", path)
            continue

        print("move", label_name, path, "->", target)

        if not os.path.exists(directory):
            os.makedirs(directory)
        shutil.copy(path, target)

    us, c = np.unique(predict_labels, return_counts=True)
    print(dict(zip([labels[u] for u in us], c)))

    with open(pklname, mode="wb") as f:
        dump(predicts, f)
