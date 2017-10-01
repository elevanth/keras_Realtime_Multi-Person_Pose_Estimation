import os
import json
import time
import cv2
import math
import numpy as np
import util
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf

from model import get_model
from inference import process
from config_reader import config_reader


batch_size = 32
keras_weights_file = "model/model.h5"
# img_root = '../dataset/ai-challenger/valid/keypoint_validation_images_20170911'
img_root = '../dataset/ai-challenger/test/keypoint_test_a_images_20170923'
# img_root = './sample_images'
img_list = os.listdir(img_root)# print('heatmap', heatmap.shape)
output_file = './results/results.json'
joint_num = 14

KERAS2AIC = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1, 2]

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

# the middle joints heatmap correpondence
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
          [55, 56], [37, 38], [45, 46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

eps = 1e-3


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

# class myEncoder(json.JSONEncoder):
#   def default(self, obj):
#     if isinstance(obj, dict):
#       return obj.__str__()
#     return json.JSONEncoder.default(self, obj)

def default(o):
    if isinstance(o, np.integer): return int(o)
    raise TypeError


def main():
    # load model
    model = get_model()
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()

    out = []

    for count in range(len(img_list)):
        # if count%1000 == 0:
        #     print('processed %d of %d'%(count, len(img_list)))

        tic = time.time()
        result = {}

        real_image_path = os.path.join(img_root, img_list[count])
        # generate image with body parts
        all_peaks, subset, _ = process(model, real_image_path, params, model_params)
        # print(all_peaks)

        result["image_id"] = img_list[count].split('.')[0]
        result["keypoint_annotations"] = {}
        human_num = len(subset)
        subset = subset.astype(dtype=np.int64)
        # print(subset.shape)

        for idx in range(human_num):
            humann_id = 'human' + str(idx+1)
            result["keypoint_annotations"][humann_id] = []
            # result["keypoint_annotations"][humann_id] = 'test'

        for idx in range(human_num):
            humann_id = 'human' + str(idx+1)
            for jn in range(joint_num):
                j_id = KERAS2AIC[jn]
                j = subset[idx][j_id-1]
                if j == -1:
                    result["keypoint_annotations"][humann_id].extend([0, 0, 0])
                else:
                    cand = all_peaks[j_id-1]
                    for c in range(len(cand)):
                        if j == cand[c][-1]:
                            result["keypoint_annotations"][humann_id].extend([cand[c][0], cand[c][1]])
                            result["keypoint_annotations"][humann_id].extend([1])

        out.append(result)
        # print(result)
        # cv2.imwrite('./results/result.jpg', canvas)

        toc = time.time()
        print ('processing the %dth image is %.5f' % (count, toc - tic))

        if count%500 == 0:
            with open(output_file, 'w') as f:
                json.dump(out, f, default=default)


if __name__ == '__main__':
    main()

