import os
import json
import argparse
import cv2
import math
import time
import numpy as np
import util
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from model import get_model

import tensorflow as tf

from inference import *

batch_size = 32
keras_weights_file = "model/model.h5"
# img_root = '../dataset/ai-challenger/valid/keypoint_validation_images_20170911'
img_root = './sample_images'
img_list = os.listdir(img_root)
output_file = './results/results.json'

JOINTS_TRANS = [13, 14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def getTensor(num):
    img_tensor = {}
    batch_list = img_list[num*batch_size : min((num+1)*batch_size, length)]
    for image_path in batch_list:
        real_image_path = os.path.join(img_root, image_path)
        image_data = tf.gfile.FastGFile(real_image_path, 'rb').read()
        img_tensor[image_path] = image_data
    return img_tensor

# with tf.Session() as sess:
#     img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg)
#     img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)

def main():


    out = []


    for img_path in img_list:

        tic = time.time()

        # load model
        model = get_model()
        model.load_weights(keras_weights_file)

        # load config
        params, model_params = config_reader()

        result = {}

        real_image_path = os.path.join(img_root, img_path)
        # generate image with body parts
        all_peaks = process(model, real_image_path, params, model_params)
        # print(all_peaks)

        result["image_id"] = img_path.split('.')[0]
        result["keypoint_annotations"] = {}
        human_num = max([len(elem) for elem in all_peaks])

        for idx in range(human_num):
            human_id = 'human' + str(idx+1)
            result["keypoint_annotations"][human_id] = []

        for ap in range(len(JOINTS_TRANS)):
            i = JOINTS_TRANS[ap]
            for j in range(human_num):
                human_id = 'human' + str(j+1)
                if j < len(all_peaks[i]):
                    result["keypoint_annotations"][human_id].extend(all_peaks[i][j][0:2])
                    result["keypoint_annotations"][human_id].extend([1])
                else:
                    result["keypoint_annotations"][human_id].extend([0, 0, 0])

        out.append(result)

        toc = time.time()
        print ('processing one image is %.5f' % (toc - tic))

    print(out)
    with open(output_file, 'w') as f:
        f.write(str(out))

if __name__ == '__main__':
    main()

