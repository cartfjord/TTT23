import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import tarfile
import imutils
import tifffile as tiff

from frcnn_train import train_polyp
from frcnn_test import parse_args, predict
from keras_frcnn import config
from keras_frcnn import resnet as nn

class DataAugmentation:
    def __init__(self):
        self.gamma_low = 0.7
        self.gamma_high = 1.3
        self.binary_threshold = 20
        self.angle_low = 0
        self.angle_high = 360
        self.affine_angle_low = -64
        self.affine_angle_high = 64
        self.alpha_min = 0.8
        self.alpha_max = 1.2
        self.beta_min = 0
        self.beta_max = 0

    def augment(self, image, label):
        image_aug, label_aug = self.rotate(image,label)
        image_aug = self.threshold(image_aug)
        #image_aug = self.brightnessAdjust(image_aug)
        #image_aug = self.threshold(image_aug)
        image_aug = self.threshold(image_aug)
        #image_aug = self.threshold(image_aug)
        image_aug, label_aug = self.perspectiveTransform(image_aug, label_aug)
        image_aug = self.gammaAdjust(image_aug, np.random.uniform(self.gamma_low, self.gamma_high))

        return image_aug, label_aug

    def rotate(self, image, label):
        angle = np.random.randint(self.angle_low, self.angle_high)

        image_rotated = imutils.rotate(image, angle)
        label_rotated = imutils.rotate(label, angle)

        return image_rotated, label_rotated

    def perspectiveTransform(self, image, label):
        rows, cols, ch = image.shape
        rand_xmin = np.random.randint(self.affine_angle_low, self.affine_angle_high)
        rand_xmax = np.random.randint(self.affine_angle_low, self.affine_angle_high)
        rand_ymin = np.random.randint(self.affine_angle_low, self.affine_angle_high)
        rand_ymax = np.random.randint(self.affine_angle_low, self.affine_angle_high)

        pts1 = np.float32([[64, 64], [cols - 64, 64 ], [64, cols - 64], [rows - 64, cols - 64]])
        pts2 = np.float32([[       64 + rand_xmin,        64 + rand_ymin],
                           [cols - 64 + rand_xmax,        64 + rand_ymin],
                           [       64 + rand_xmin, cols - 64 + rand_ymax],
                           [rows - 64 + rand_xmax, cols - 64 + rand_ymax]])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        perspective_image = cv2.warpPerspective(image, M, (384, 288))
        perspective_label = cv2.warpPerspective(label, M, (384, 288))

        return perspective_image, perspective_label

    def threshold(self, image):
        ret, thresh_image = cv2.threshold(image, self.binary_threshold, 255, cv2.THRESH_TOZERO)
        return thresh_image

    def gammaAdjust(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def brightnessAdjust(self, image):
        alpha = np.random.uniform(self.alpha_min, self.alpha_max)

        image = image*alpha
        return cv2.convertScaleAbs(image, alpha)