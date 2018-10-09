import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
import cv2
import tarfile
import DataAugmentation
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
        if (filename.endswith('.tif')):
            images.append(tiff.imread(os.path.join(path, filename)))
            #print(images[-1].shape)
        else:
            images.append(cv2.imread(os.path.join(path, filename)))
            #print(images[-1].shape)
    print('Done!\n')
    return images

def convertImages(path):
    files = os.listdir(path)
    for filename in files:
        new_filename = filename.replace('tif', 'png')
        image = tiff.imread(os.path.join(path, filename))
        print(image)

        matplotlib.image.imsave(os.path.join(path, new_filename), image)

def showImage(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

def showImageOverlay(image, label):
    plt.figure()
    plt.imshow(image)
    plt.imshow(label, alpha=0.3)
    plt.show()

#path = 'model/PolypModel.hdf5.tar.gz'
def extractTarGz(file):
    if (os.path.exists(file)):
        if (file.endswith("tar.gz")):
            tar = tarfile.open(file, "r:gz")
            tar.extractall()
            tar.close()

def generateTrainingFormat(labels, path, filename):
    files = os.listdir(path)
    assert(len(files) == len(labels))
    f = open(filename, 'w')

    print(type(np.array(labels[0])))

    for i in range(0, len(labels)):
        #_, contours, _ = cv2.findContours(np.array(labels[i]), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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

    train_labels = [cv2.cvtColor(lbl, cv2.COLOR_BGR2GRAY) for lbl in train_labels]
    #print("shape:", train_labels[-1].shape)


    for i in range(0, len(train_labels)):
        _, train_labels[i] = cv2.threshold(train_labels[i], 50, 255, cv2.THRESH_BINARY)



    test_images = loadImages(cfg.test_images_path)
    test_labels = loadImages(cfg.test_labels_path)
    test_labels = [cv2.cvtColor(lbl, cv2.COLOR_BGR2GRAY) for lbl in test_labels]

    Augmentor = DataAugmentation.DataAugmentation()
    cv2.imshow("Original_label", train_labels[0])

    augmented_images = []
    augmented_labels = []

    image_names = os.listdir("./data_png/train/image")
    label_names = os.listdir("./data_png/train/label")

    print("Type", type(image_names))
    for i in range(0, len(train_images)):
        print("saving image: ", i)
        for j in range(0, 3):
            augmented_images_1, augmented_labels_1 = Augmentor.augment(train_images[i], train_labels[i])
            augmented_images.append(augmented_images_1)
            augmented_labels.append(augmented_labels_1)
            matplotlib.image.imsave("./data_png/train_augmented/image/"+image_names[i][:-4]+"_aug_"+str(j)+".png", cv2.cvtColor(augmented_images_1, cv2.COLOR_BGR2RGB))
            matplotlib.image.imsave("./data_png/train_augmented/label/"+label_names[i][:-4]+"_aug_"+str(j)+".png", augmented_labels_1)
            #scipy.misc.imsave("")






    cv2.imshow("26_augmented_1", augmented_images[0])
    cv2.imshow("26_augmented_2", augmented_images[1])
    cv2.imshow("26_augmented_3", augmented_images[2])

    cv2.imshow("26_augmented_label_1", augmented_labels[0])
    cv2.imshow("26_augmented_label_2", augmented_labels[1])
    cv2.imshow("26_augmented_label_3", augmented_labels[2])


    #showImageOverlay(train_images[0], train_labels[0])
    """
    generateTrainingFormat(train_labels, cfg.train_labels_path, cfg.simple_label_file)


    ##########################################################################################
    # Configuration parameters that can be changed.
    cfg.num_epochs = 600
    cfg.epoch_len = 100
    cfg.use_horizontal_flips = True
    cfg.use_vertical_flips = True
    cfg.rot_90 = True
    cfg.num_rois = 8
    ##########################################################################################

    cfg.base_net_weights = os.path.join('./model/', nn.get_weight_path()) #What is this? Pretrained model?
    #It looks for './model/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

    ## Train the model
    #train_polyp(cfg)

    ##
    #predict(cfg)
    predict(parse_args(cfg))

#    cv2.rectangle(train_images[328],(49,176),(121,238), (0,0,255), 3)
#    cv2.rectangle(train_images[328],(187,134),(231,178), (0,0,255), 3)

 #   showImageOverlay(train_images[328],train_labels[328])

    #showImageOverlay(train_images[8], train_labels[8])
    """
main()
