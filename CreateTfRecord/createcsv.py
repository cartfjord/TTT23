import os, cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def generateTrainingFormat():

    path = '../data_png/train/label/' 
    #path = '../data_png/test_all/label/'

    files = os.listdir(path)
    
    filename = 'train.csv'
    f = open(filename, 'w')
    f.write("class,fileName,height,width,xmax,xmin,ymax,ymin\n")

    for image in files:
        img = cv2.imread(os.path.join(path, image), 0)
        _, img = cv2.threshold(img, 50, 255,cv2.THRESH_BINARY )
        _, contours, _ = cv2.findContours(np.array(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x1,y1,w,h = cv2.boundingRect(cnt)
            area = h*w
            if area > 25:
                x2 = x1 + w
                y2 = y1 + h
                f.write("polyp" + "," + image + "," + str(h) + "," + str(w) + "," + str(x2) + "," + str(x1) + "," + str(y2) + "," + str(y1) + "\n")
                #cv2.rectangle(img, (x1,y1), (x2,y2), (50,50,50) ,5)
                #cv2.imshow('image',img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

    f.close()


generateTrainingFormat()


