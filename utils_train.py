import torch
import torch.nn.functional as F
import numpy as np
from datasets import *
from torchvision import transforms
import cv2

def calc_iou(bb1, bb2): #[left_top_x, left_top_y, right_bottom_x, right_bottom_y], bb1.shape: [1, 4], bb2.shape: [N, 4]
    inter_x1 = np.maximum(bb1[:,0], bb2[:,0])
    inter_y1 = np.maximum(bb1[:,1], bb2[:,1])
    inter_x2 = np.minimum(bb1[:,2], bb2[:,2])
    inter_y2 = np.minimum(bb1[:,3], bb2[:,3])

    inter_area = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None) * np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    union_area = (bb1[:, 2] - bb1[:, 0]) * (bb1[:, 3] - bb1[:, 1]) + (bb2[:, 2] - bb2[:, 0]) * (bb2[:, 3] - bb2[:, 1]) - inter_area

    return inter_area / union_area

def neg_anchors_for_single_scale(label, anchors, grid_size=13, inp_dim=416, thresh=0.5):
    label_cls, label_bbx = label
    bbx_num = grid_size ** 2 * len(anchors)
    stride = inp_dim // grid_size

    idx = np.arange(bbx_num) // len(anchors)
    y_c = (idx // grid_size).astype(np.float32) * stride 
    x_c = (idx % grid_size).astype(np.float32) * stride

    anchors = np.tile(np.array(anchors), (grid_size * grid_size, 1))
    bbox_anchors = np.zeros((bbx_num, 4))
    bbox_anchors[:, 0] = x_c - anchors[:, 0] / 2
    bbox_anchors[:, 1] = y_c - anchors[:, 1] / 2
    bbox_anchors[:, 2] = x_c + anchors[:, 0] / 2
    bbox_anchors[:, 3] = y_c + anchors[:, 1] / 2

    neg = []
    for bbx in label_bbx:
        bbx = bbx[None, ...]
        ious = calc_iou(bbx, bbox_anchors)
        neg.append(ious < thresh)

    neg = np.all(neg, axis=0)

    return neg

def neg_anchors(label, anchors, grid_sizes=[13, 26, 52], inp_dim=416, thresh=0.5):
    negs = [neg_anchors_for_single_scale(label, anchors[i], grid_sizes[i], inp_dim, thresh) for i in range(len(grid_sizes))]

    return negs

def pos_anchors(label, anchors, grid_sizes=[13, 26, 52], inp_dim=416, thresh=0.5):
    label_cls, label_bbx = label
    anchors = np.array(anchors)
    x_c, y_c = (label_bbx[:, 0] + label_bbx[:, 2]) / 2, (label_bbx[:, 1] + label_bbx[:, 3]) / 2

    pos = []
    for i, bbx in enumerate(label_bbx):
        bbx = bbx[None, ...]

        pos_grid = []
        for j, grid_size in enumerate(grid_sizes):
            anchors_grid = anchors[j]
            bbox_anchors = np.zeros((len(anchors), 4))
            stride = inp_dim // grid_size
            x_idx, y_idx = int(x_c[i] / stride), int(y_c[i] / stride)
            bbox_anchors[:, 0] = x_idx * stride - anchors_grid[:, 0] / 2
            bbox_anchors[:, 1] = y_idx * stride - anchors_grid[:, 1] / 2
            bbox_anchors[:, 2] = x_idx * stride + anchors_grid[:, 0] / 2
            bbox_anchors[:, 3] = y_idx * stride + anchors_grid[:, 1] / 2

            ious = calc_iou(bbx, bbox_anchors)
            idx_max = np.argmax(ious)
            val_max = np.max(ious)

            pos_grid.append((j, idx_max, val_max, x_idx, y_idx))  # scale_idx, anchor_idx, max_iou, x_idx, y_idx

        pos_grid.sort(key=lambda x: x[2], reverse=True)
        pos.append(pos_grid[0])

    return pos

def pos_neg_anchors(label, anchors, grid_sizes=[13, 26, 52], inp_dim=416, thresh=0.5):
    pos = pos_anchors(label, anchors, grid_sizes=grid_sizes, inp_dim=inp_dim, thresh=thresh)
    neg = neg_anchors(label, anchors, grid_sizes=grid_sizes, inp_dim=inp_dim, thresh=thresh)
    for p in pos:
        scale_idx, anchor_idx, _, x_idx, y_idx = p
        neg[scale_idx][y_idx*len(anchors)*grid_sizes[scale_idx] + x_idx*len(anchors) + anchor_idx] = False

    return pos, neg


if __name__ == '__main__':
    dtset = PascalVosDataset('/data/liuf/PascalVoc/voc_train.txt', transforms.Compose([Rescale(416), RandomHorizontalFlip(1), BottomRightPad(416)]))
    anchors = [ [[116,90], [156,198], [373,326]],  [[30,61], [62,45], [59,119]],  [[10,13], [16,30], [33,23]] ]  # yolov3

    sample = dtset[11]
    img, label = sample['image'], sample['label']
    label_cls, label_bbx = label
    label_bbx = label_bbx.astype(np.int32)
    for i, bb in enumerate(label_bbx):
        point1 = (int(bb[0]), int(bb[1]))
        point2 = (int(bb[2]), int(bb[3]))
        cv2.rectangle(img, point1, point2, [255, 0, 0], 1)
        text = voc_classes[label_cls[i]]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6 , 1)[0]
        cv2.rectangle(img, point1, (point1[0]+text_size[0]+2, point1[1]+text_size[1]+2), [255, 0, 0], -1)
        cv2.putText(img, text, (point1[0]+1, point1[1]+text_size[1]+1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, [225,255,255], 1)

    cv2.imwrite('test_util1.jpg', img)
    
    pos, neg = pos_neg_anchors(label, anchors=anchors)
    print(pos)
    k=0
    ii = 0
    for i, n in enumerate(neg[k]):
        if not n:
            ii+=1
            g = 13 * 2**k; s = 32 // 2**k
            x_c = (i % (g * 3)) // 3 * s
            y_c = (i // (g*3)) * s
            an_i = i % (g * 3) % 3

            anc = anchors[k][an_i]
            print(x_c // s, y_c // s, anc)
            point1 = int(x_c - anc[0] / 2), int(y_c - anc[1] / 2)
            point2 = int(x_c + anc[0] / 2), int(y_c + anc[1] / 2)
            # if ii == 5: cv2.rectangle(img, point1, point2, [0, 0, 255], 1)
            cv2.rectangle(img, point1, point2, [0, 0, 255], 1)

    cv2.imwrite('test_util2.jpg', img)




