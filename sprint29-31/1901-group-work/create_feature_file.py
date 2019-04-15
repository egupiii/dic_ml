#!/usr/bin/env python
# ! -*- coding: utf-8 -*-
import numpy as np
from time import sleep
import subprocess
import os
#import pygame.mixer
import cv2
import dummy_detection
import dummy_prediction
import dummy_prediction2
import detection
import prediction

#from PIL import Image

#ファイルパス
KAGO_FILE_PATH = "./sys/kago.csv"
PRICE_FILE_PATH = "./etc/price_list.csv"
PRICE_FILE_PATH = "./etc/price_list.csv"
SCANNED_IMAGE_FILE_PATH = "./sys/scanned_image.jpg"
PLEASE_SCAN_FILE_PATH = "./sys/please_scan.jpg"
TEST_SCAN_FILE_PATH = "./sys/test_scanned_image.jpg"
WINDOW_NAME = 'Scanned item'

def display_scanned_item(scanned_image):
    #save SCANNED_IMAGE_FILE_PATH
    cv2.imwrite(SCANNED_IMAGE_FILE_PATH, scanned_image)

    return

def display_scanned_item2(scanned_image):
    #save SCANNED_IMAGE_FILE_PATH
    #cv2.destroyAllWindows()
    #cv2.waitKey(1)

    cv2.imshow(WINDOW_NAME, scanned_image)
    cv2.waitKey(1)
    return

def display_message(message):
    cmd = "cp {} {}".format(message, SCANNED_IMAGE_FILE_PATH)
    try:
        os.popen(cmd)
    except:
        print("Command execution error. {}".format(cmd))

    return


def update_price_file(label, name, price):
    ret = True
    with open(PRICE_FILE_PATH, mode='a+') as f:
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


# Main Loop
if __name__ == '__main__':

    #初期化　モジュールのインスタを作る
    #c_detect = dummy_detection.DummyDetection(False)
    c_detect = detection.detection()
    #c_predict = dummy_prediction.DummyPrediction(False)
    c_predict = prediction.prediction(5)

    res = subprocess.check_call('clear')
    display_message(PLEASE_SCAN_FILE_PATH)

    #Main loop
    while True:
        #検出部
        scanned_image, padded_image, image_w_bounding = c_detect.object_detection()
        #scanned_image = c_detect.detection()
        if scanned_image is None:
            #sleep(1)
            continue

        #scanned_image = cv2.imread(TEST_SCAN_FILE_PATH)
        display_scanned_item2(scanned_image)
        #display_message(TEST_SCAN_FILE_PATH)

        #Input product name, label & price
        item_name = input("Please set product name\n")
        item_label = input("Please set product label\n")
        item_price = input("Please set product price\n")

        padded_image = padded_image[np.newaxis]
        c_predict.create_feature_from_image(padded_image, item_label)
        update_price_file(item_label, item_name, item_price)

        #Detecitonのmodeをfalseに変える
        c_detect.set_mode(False)

        display_message(PLEASE_SCAN_FILE_PATH)


