import re

# https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
def get_iou(bbP, bbL):
    # [x1, y1, x2, y2]
    bb1 = [bbP[1], bbP[0], bbP[3], bbP[2]]
    bb2 = bbL

    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def printDict(d):
    for img in d:
        print(d[img])

def parse(bbDetectedFile):
    d = {}

    f_detected = open(bbDetectedFile, 'r')
    for line in f_detected:
        a = re.findall('\d+\.png', line)
        if (len(a) > 0):
            img = a[0]
            d[img] = {'detected':[], 'label': []}
        else:
            b = re.findall('\d+.\d+', line)
            b = list(map(float, b))
            b = [round(x) for x in b]
            d[img]['detected'].append(b)
    printDict(d)


def create_bb_dict(bbLabelFile, bbDetectedFile):
    d = {}

    f_detected = open(bbDetectedFile, 'r')
    for line in f_detected:
        a = re.findall('\d+\.png', line)
        if (len(a) > 0):
            img = a[0]
            d[img] = {'detected':[], 'label': []}
        else:
            b = re.findall('\d+.\d+', line)
            b = list(map(float, b))
            b = [round(x) for x in b]
            d[img]['detected'].append(b)
    f_detected.close()

    f_labels = open(bbLabelFile, 'r')
    for line in f_labels:
        a = re.findall('\d+\.png|\d+', line)
        img = a[0]
        bbLabels = list(map(int, a[1:5]))

        if(img not in d):
            #print("Image {} not found in results".format(img))
            d[img] = {'detected':[], 'label': []}
            d[img]['label'].append(bbLabels)
        else:
            d[img]['label'].append(bbLabels)

    f_labels.close()
    return d

def calculatePositives(bbLabelFile, bbDetectedFile):
    iou_threshould = 0.5
    tp = 0;
    fp = 0;
    fn = 0;
    rm_d = []
    rm_l = []

    d = create_bb_dict(bbLabelFile, bbDetectedFile)

    result = {'TP':None, 'FP':None, 'FN':None}

    for img in d:
        for detected in d[img]['detected']:
            for label in d[img]['label']:
                iou = get_iou(detected, label)
                if(iou > iou_threshould):
                    if(detected not in rm_d):
                        rm_d.append(detected)
                    if(label not in rm_l):
                        rm_l.append(label)
                    tp += 1
        for rm in rm_d:
            d[img]['detected'].remove(rm)
        for rm_ in rm_l:
            d[img]['label'].remove(rm_)
        rm_d = []
        rm_l = []

    for img in d:
        fp+=len(d[img]['detected'])
        fn+=len(d[img]['label'])

    result['TP'] = tp
    result['FP'] = fp
    result['FN'] = fn

    print(result)
    return result

def calculatePerformanceMetrics(predictionResults):
    tp = predictionResults['TP']
    fp = predictionResults['FP']
    fn = predictionResults['FN']

    pre = tp/(tp+fp)
    rec = tp/(tp+fn)

    print('Precision: {0:.2f}'.format(pre))
    print('Recall: {0:.2f}'.format(rec))

def evaluation():
    labelFile = 'test1.csv'
    predictionFile = 'final_10000_08.txt'

    print()
    print('Prediction file: {}'.format(predictionFile))
    predictionResults = calculatePositives(labelFile, predictionFile)
    calculatePerformanceMetrics(predictionResults)
    print()

evaluation()
