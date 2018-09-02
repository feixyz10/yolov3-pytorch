from utils import *
from models import Yolo3Tiny
import cv2
import torch

img_orig = cv2.imread('data/dog.jpg')
img = cv2.resize(img_orig, (416, 416))
img = img[:,:,::-1].transpose((2, 0, 1)) / 255.0
img = torch.FloatTensor(img).unsqueeze(0)

yolo3 = Yolo3Tiny([[[81,82], [135,169], [344,319]], [[10,14], [23,27], [37,58]]])
yolo3.load_state_dict(torch.load('weights/yolov3_tiny.pth'))

yolo3.train(False)
y1, y2 = yolo3(img)
print(y1.shape, y2.shape)

y1 = transform_predictions(y1, yolo3.anchors[0], 416)
y2 = transform_predictions(y2, yolo3.anchors[1], 416)

y = torch.cat([y1, y2], 1).squeeze_(0)
y = preproc_before_nms(y, dim=416, thres=0.6)
pred = nms(y, 0.5)
print(pred)

cls_names = [x for x in open('coco.names.txt', 'r').read().split('\n') if len(x) > 0]

img_pred = draw_bboxes(img_orig, pred, cls_names)
cv2.imshow('image', img_pred)
cv2.waitKey(0)
