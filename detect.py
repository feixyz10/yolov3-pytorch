from utils_detect import *
from models import Yolo3Tiny, Yolo3
import cv2
import torch
import argparse

parser = argparse.ArgumentParser(description="Yolov3")
parser.add_argument('--model', default='yolov3', type=str)
parser.add_argument('--weight', default='weights/yolov3.pth', type=str)
parser.add_argument('--image', default='data/street.jpg', type=str)
parser.add_argument('--thres', default=0.5, type=float)
args = parser.parse_args()

assert args.model == 'yolov3' or args.model == 'yolov3-tiny', "Only 'yolov3' and 'yolov3-tiny are available now!"

anchors1 = [ [[116,90], [156,198], [373,326]],  [[30,61], [62,45], [59,119]],  [[10,13], [16,30], [33,23]] ]  # yolov3
anchors2 = [ [[81,82], [135,169], [344,319]],  [[10,14], [23,27], [37,58]] ] # yolov3-tiny
if args.model == 'yolov3':
    yolo3 = Yolo3(anchors1)
elif args.model == 'yolov3-tiny':
    yolo3 = Yolo3Tiny(anchors2)
yolo3.load_state_dict(torch.load(args.weight))

img_orig = cv2.imread(args.image)
img = cv2.resize(img_orig, (416, 416))
img = img[:,:,::-1].transpose((2, 0, 1)) / 255.0
img = torch.FloatTensor(img).unsqueeze(0)

yolo3.train(False)
y = yolo3(img)

y = [transform_predictions(y_, yolo3.anchors[i], 416) for i, y_ in enumerate(y)]
# y = [y[0]]

y = torch.cat(y, 1).squeeze_(0)
try:
    y = preproc_before_nms(y, dim=416, thres=args.thres)
    pred = nms(y, 0.5)
except:
    pred = []

cls_names = [x for x in open('coco.names.txt', 'r').read().split('\n') if len(x) > 0]

print('%d objects are detected.'%len(pred))
for p in pred:
    name = cls_names[int(p[0])]
    conf = int(p[1] * 100)
    print(name, '%d%%'%conf, sep=': ')

img_pred = draw_bboxes(img_orig, pred, cls_names)
cv2.imshow('image', img_pred)
cv2.imwrite(args.image.split('.')[0]+'_prediction.jpg', img_pred)
cv2.waitKey(0)
