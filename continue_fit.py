# -*- coding: utf-8 -*-
"""
訓練を中断-継続するためのモジュール。
Tensorboardのlogディレクトリにpickleファイルを入れる前提なので、
Tensorboardと一緒に使う事。
使い方はこんな感じ。

import continue_fit as cf
model.fit_generator(
        train_generator,
        epochs=epochs,
        initial_epoch=cf.load_epoch_init(model_name),
        validation_data=validation_data,
        callbacks=[
            TensorBoard(model_name),
            cf.early_stopping(model, model_name),
            ])
"""
from keras.callbacks import LambdaCallback
import pickle
import os
from functools import reduce
from keras import Model


def concat(lst)->list:
    return reduce(lambda x, y: x+y, lst)


def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def load_pickle(path):
    if os.path.exists(path):
        with open(path, mode="rb") as f:
            return pickle.load(f)
    else:
        return None


def early_stopping(model: Model, name: str):

    if os.path.exists(name+".h5"):
        model.load_weights(name+".h5")

    def onend(epoch, logs):
        model.save(name+".h5")
        with open(name+"/epochs", mode="wb") as f:
            pickle.dump(epoch+1, f)
        val_loss = load_pickle(name+"/val_loss")

        if val_loss is None:
            val_loss = 1e10

        if val_loss > logs["val_loss"]:
            model.save(name + "best.h5")

            with open(name+"/val_loss", mode="wb") as f:
                pickle.dump(logs["val_loss"], f)

            with open(name+"/best_epochs", mode="wb") as f:
                pickle.dump(epoch, f)

    return LambdaCallback(on_epoch_end=onend)


def load_epoch_init(logpath: str="./logs") -> int:
    epoch_init = load_pickle(logpath+"/epochs")

    if epoch_init is None:
        epoch_init = 0

    return epoch_init
