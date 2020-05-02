import cv2
from os.path import isfile, isdir

# class2id_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7,
#                          'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
#                          'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
#
# id2class_dict = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat',
#                  8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
#                  15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}

boxes_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

my_box_cv_colors = [[200, 200, 200],        # aeroplane (silver)
                    [50, 150, 250],         # bicycle (turquoise)
                    [0, 255, 0],            # bird (green)
                    [0, 0, 255],            # boat (blue)
                    [255, 255, 255],        # bottle (white)
                    [255, 0, 255],          # bus (pink)
                    [255, 0, 0],            # car (red)
                    [50, 0, 100],           # cat (dark purple)
                    [50, 25, 0],            # chair(brown)
                    [100, 100, 0],          # cow (milk brown)
                    [100, 0, 0],            # diningtable (dark red)
                    [100, 0, 200],          # dog (purple)
                    [200, 100, 0],          # horse (dark orange)
                    [0, 80, 150],           # motorbike (dark turquoise)
                    [255, 255, 0],          # person (yellow)
                    [128, 255, 0],          # pottedplant (light green)
                    [0, 100, 0],            # sheep (dark green)
                    [250, 150, 200],        # sofa (light pink)
                    [50, 50, 50],           # train (dark gray)
                    [0, 0, 100]]            # tvmonitor (dark blue)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.3


