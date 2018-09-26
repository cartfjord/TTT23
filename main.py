import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

import tifffile as tiff

from frcnn_train import train_polyp
from frcnn_test import parse_args, predict
from keras_frcnn import config
from keras_frcnn import resnet as nn

def loadImages(path):
    files = os.listdir(path)
    print('Loading',len(files), 'images from', path)
    images = []

    for filename in files:
        images.append(tiff.imread(os.path.join(path, filename)))
    print('Done!\n')
    return images

def showImage(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

def showImageOverlay(image, label):
    plt.figure()
    plt.imshow(image)
    plt.imshow(label, alpha=0.3)
    plt.show()


def generateTrainingFormat(labels, path, filename):
    files = os.listdir(path)
    assert(len(files) == len(labels))
    f = open(filename, 'w')
    
    for i in range(0, len(labels)):
        _, contours, _ = cv2.findContours(np.array(labels[i]), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #contours = sorted(
        # contours, key = cv2.contourArea, reverse = True)[:10]
        for cnt in contours:
            x1,y1,w,h = cv2.boundingRect(cnt)
            area = h*w
            if area > 25:
                x2 = x1 + w
                y2 = y1 + h
                f.write(path+"/"+files[i]+","+str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+"Polyp\n")
    
    f.close()


def main():
    cfg = config.Config()

    train_images = loadImages(cfg.train_images_path)
    train_labels = loadImages(cfg.train_labels_path)

    test_images = loadImages(cfg.test_images_path)
    test_labels = loadImages(cfg.test_labels_path)

    generateTrainingFormat(train_labels, cfg.train_labels_path, cfg.simple_label_file)


    ##########################################################################################
    # Configuration parameters that can be changed.
    cfg.num_epochs = 10
    cfg.epoch_len = 100
    cfg.use_horizontal_flips = True
    cfg.use_vertical_flips = True
    cfg.rot_90 = True
    cfg.num_rois = 32
    ##########################################################################################

    cfg.base_net_weights = os.path.join('./model/', nn.get_weight_path()) #What is this? Pretrained model?
    #It looks for './model/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

    ## Train the model
    #train_polyp(cfg)

    ##
    #predict(cfg)
    predict(parse_args(cfg))

    cv2.rectangle(train_images[328],(49,176),(121,238), (0,0,255), 3)
    cv2.rectangle(train_images[328],(187,134),(231,178), (0,0,255), 3)

    showImageOverlay(train_images[328],train_labels[328])

    showImageOverlay(train_images[8], train_labels[8])
    
main()
