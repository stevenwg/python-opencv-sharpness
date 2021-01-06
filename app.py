# Created by Chao-Chuan Lu on 2020/11/11.

import os
import glob
import numpy as np
import cv2
import math
import timeit
import logging.config
import progressbar # pip install progressbar2
import matplotlib.pyplot as plt

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log', 'logging_config.ini')
logging.config.fileConfig(log_file_path)
logger = logging.getLogger('ChaoChuanLu')

def calcSharpness(image):
    """Calculate the image's sharpness with FFT. The smaller value means more sharpness.
    
    Args:
        image: testing image.
        
    Returns:
        FFT: The smaller value means more sharpness.
    
    """
    grad_x = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3, scale=1, delta=0)
    grad_y = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3, scale=1, delta=0)
    norm_x = cv2.norm(grad_x)
    norm_y = cv2.norm(grad_y)
    # cv2.namedWindow('grad_x', cv2.WINDOW_NORMAL)
    # cv2.imshow('grad_x', grad_x)
    # cv2.namedWindow('grad_y', cv2.WINDOW_NORMAL)
    # cv2.imshow('grad_y', grad_y)
    # cv2.waitKey(1)
    sum = norm_x * norm_x + norm_y * norm_y
    result = sum / image.size
    return result

if __name__ == '__main__':
    # logger.info('========== Calculate Image Sharpness ==========')
    # image = cv2.imread('data/sharpness_barpattern_blur.jpg')
    # sharpness = calcSharpness(image)
    # print('Sharpness: ', sharpness)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    
    logger.info('========== Calculate Video Sharpness ==========')
    cap = cv2.VideoCapture('data/IMG_5111.MOV')
    sharpness_list = []
    bar = progressbar
    print('Total Frames: ', cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in bar.progressbar(range(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if ret is False:
            break
        frame = cv2.rotate(frame, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        sharpness = round(calcSharpness(frame))
        sharpness_list.append(sharpness)
        
        if sharpness < 600 or sharpness > 2000:
            print('Sharpness: ', sharpness)
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
        # print('Sharpness: ', sharpness)
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()

    # print(sharpness_list)
    sharpness_list_count = {i:sharpness_list.count(i) for i in np.unique(sharpness_list)}
    plt.bar(sharpness_list_count.keys(), sharpness_list_count.values())
    plt.show()