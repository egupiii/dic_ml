from time import sleep
#from PIL import Image
import numpy as np
#import json
import os
import subprocess
import sys
import select
import glob
import pickle

import dummy_prediction


"""
フォルダにあるimageを読み込む
Fileの名前から商品名、ラベル、価格を取得 (NAEM_LABEL_PRICE.jpg)
データをtraining関数に渡す
training関数から帰ってきた結果（抽出された特徴量）をファイルに保存
ファイルはpickle形式, labelと配列を紐付ける
"""

#Item imageフォルダパス
ITEM_IMAGE_PATH = "./item_image/"

#Priceファイルパス
PRICE_FILE_PATH = "./etc/"

#特徴量ファイルのフォルダパス
FEATURE_FILE_PATH = "./feature/"


def read_feature(path, label):
    file_path = "{}_feature.dump".format(path + label)
    with open(file_path, 'rb') as f:
        feature = pickle.load(f)

    return feature


def get_files(path):
    yield [os.path.abspath(p) for p in glob.glob(path)]


def update_feature_file(label, data):
    file_path = "{}_feature.dump".format(FEATURE_FILE_PATH + label)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    return


def update_price_file(label, name, price):
    ret = True
    path = "{}/price_list.csv".format(PRICE_FILE_PATH)
    with open(path, mode='a+') as f:
        f.seek(0)
        next_index = 1
        for i, line in enumerate(f.readlines()):
            next_index = next_index + 1
            line = line.rstrip('\n')
            tmp = line.split(",")
            if tmp[1] == label:
                ret = False
                break

        #すでにリストある商品は追記しない
        if ret:
            new_line = str("{},{},{},{}\n".format(next_index, label, name, price))
            f.write(new_line)

    return


def call_training_func(model, image_dir, label):
    feature = model.train(image_dir, label)

    return feature


# Main Loop
if __name__ == '__main__':

    #初期化
    c_predict = dummy_prediction.DummyPrediction(False)

    #ITEM_IMAGE_PATHからファイルを読み出し特徴量抽出＋Priceリストに商品、ラベル、価格を追記
    #features = np.array()
    for dir_name in os.listdir(ITEM_IMAGE_PATH):
        if os.path.isfile(ITEM_IMAGE_PATH + dir_name):
            continue

        tmp = dir_name.split("_")
        item_name = tmp[0]
        item_label = tmp[1]
        item_price = tmp[2]
        #item_image = Image.open(file_name)
        item_feature = call_training_func(c_predict, dir_name, item_label)

        #特徴量の保存
        update_feature_file(item_label, item_feature)

        #priceファイルの更新
        update_price_file(item_label, item_name, item_price)

        print("Done {}".format(dir_name))


