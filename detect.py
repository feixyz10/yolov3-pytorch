from utils import *
from models import Yolo3Tiny, Yolo3
import cv2
import torch

anchors1 = [[[10,13],  [16,30],  [33,23]],  [[30,61],  [62,45],  [59,119]],  [[116,90],  [156,198],  [373,326]]]  # yolov3
anchors2 = [[[10,14],  [23,27],  [37,58]],  [[81,82],  [135,169],  [344,319]]] # yolov3-tiny

img_orig = cv2.imread('data/person.jpg')
img = cv2.resize(img_orig, (416, 416))
img = img[:,:,::-1].transpose((2, 0, 1)) / 255.0
img = torch.FloatTensor(img).unsqueeze(0)

yolo3 = Yolo3(anchors1)
yolo3.load_state_dict(torch.load('weights/yolov3.pth'))

yolo3.train(False)
y = yolo3(img)

y = [transform_predictions(y_, yolo3.anchors[2-i], 416) for i, y_ in enumerate(y)]

y = torch.cat(y, 1).squeeze_(0)
try:
    y = preproc_before_nms(y, dim=416, thres=0.5)
    pred = nms(y, 0.25)
    print(pred, pred.shape)
except:
    pred = []

cls_names = [x for x in open('coco.names.txt', 'r').read().split('\n') if len(x) > 0]

img_pred = draw_bboxes(img_orig, pred, cls_names)
cv2.imshow('image', img_pred)
cv2.waitKey(0)
