import keras
import matplotlib.pyplot as plt
import scipy 
import os
import numpy as np
from PIL import Image
import cv2

import tifffile as tiff

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
                f.write("."+path+"/"+files[i]+","+str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+"Polyp\n")
    
    f.close()





def main():
    train_images = loadImages('./data/train/image')
    train_labels = loadImages('./data/train/label')

    test_images = loadImages('./data/test/image')
    test_labels = loadImages('./data/test/label')

    generateTrainingFormat(train_labels, './data/train/label', 'BoundingBoxesTrain.txt')

    cv2.rectangle(train_images[328],(49,176),(121,238), (0,0,255), 3)
    cv2.rectangle(train_images[328],(187,134),(231,178), (0,0,255), 3)

    showImageOverlay(train_images[328],train_labels[328])

    showImageOverlay(train_images[8], train_labels[8]);
    
main()
