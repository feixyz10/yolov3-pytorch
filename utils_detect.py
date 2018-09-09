import torch
import torch.nn.functional as F
import numpy as np
import cv2

def transform_dimensions(y, num_anchors):
    batch_size = y.size(0)
    grid_size = y.size(2)
    y = y.view(batch_size, -1, grid_size*grid_size)
    y = y.transpose(1, 2).contiguous()
    y = y.view(batch_size, grid_size*grid_size*num_anchors, -1)

    return y

def transform_predictions(y, anchors, inp_dim): # y.shape: [B, 255, grid, grid]
    num_anchors = len(anchors)
    grid_size = y.size(2)
    y = transform_dimensions(y, num_anchors) # y.shape: [B, N, 85]
    bb_size = y.size(1)

    idx = np.arange(bb_size) // len(anchors)
    y_idx = (idx // grid_size).astype(np.float32)
    x_idx = (idx % grid_size).astype(np.float32)

    x_idx = torch.FloatTensor(x_idx, device=y.device)
    y_idx = torch.FloatTensor(y_idx, device=y.device)

    y[:, :, 0].sigmoid_().add_(x_idx)
    y[:, :, 1].sigmoid_().add_(y_idx)

    dim = torch.FloatTensor(anchors, device=y.device)
    dim = dim.repeat(grid_size*grid_size, 1).unsqueeze(0)

    y[:, :, 2:4].exp_().mul_(dim)
    y[:, :, 4:].sigmoid_()

    stride = inp_dim // grid_size
    y[:, :, :2] *= stride

    return y

def preproc_before_nms(y, dim=416, thres=0.5): # y.shape: [N, 85], for one image only
    y[:, 5:].mul_(y[:, 4:5])
    max_conf, max_conf_idx = torch.max(y[:,5:], 1)
    max_conf.unsqueeze_(1)
    max_conf_idx = max_conf_idx.float().unsqueeze(1)
    left_top_x = torch.clamp(y[:, 0] - y[:, 2]/2, min=0.0).unsqueeze(1)
    left_top_y = torch.clamp(y[:, 1] - y[:, 3]/2, min=0.0).unsqueeze(1)
    right_bot_x = torch.clamp(y[:, 0] + y[:, 2]/2, max=dim-1.0).unsqueeze(1)
    right_bot_y = torch.clamp(y[:, 1] + y[:, 3]/2, max=dim-1.0).unsqueeze(1)

    y = torch.cat([max_conf_idx, max_conf, left_top_x, left_top_y, right_bot_x, right_bot_y], 1)
    mask = y[:, 1] > thres

    # y_, _ = torch.sort(y[:, 1], descending=True)
    # print(y_[:10])

    return y[mask]

def calc_iou(bb1, bb2): #[left_top_x, left_top_y, right_bottom_x, right_bottom_y], bb1.shape: [1, 4], bb2.shape: [N, 4]
    inter_x1 = torch.max(bb1[:,0], bb2[:,0])
    inter_y1 = torch.max(bb1[:,1], bb2[:,1])
    inter_x2 = torch.min(bb1[:,2], bb2[:,2])
    inter_y2 = torch.min(bb1[:,3], bb2[:,3])

    inter_area = torch.clamp(inter_x2-inter_x1, min=0) * torch.clamp(inter_y2-inter_y1, min=0)
    union_area = (bb1[:, 2] - bb1[:, 0]) * (bb1[:, 3] - bb1[:, 1]) + (bb2[:, 2] - bb2[:, 0]) * (bb2[:, 3] - bb2[:, 1]) - inter_area

    return inter_area / union_area


def nms(y, iou_thres): # y.shape: [N, 85], for one image only
    _, sort_idx = torch.sort(y[:, 1], descending=True)
    y = y[sort_idx]
    classes = torch.unique(y[:, 0])
    
    y_res = None
    flag = True
    for cls in classes:
        y_cls = y[y[:,0]==cls]
        bb_size = y_cls.size(0)
        while bb_size > 0:
            if flag:
                y_res = y_cls[0:1, :]
                flag = False
            else:
                y_res = torch.cat([y_res, y_cls[0:1,:]], 0)

            if bb_size == 1: break
            ious = calc_iou(y_cls[0:1, 2:], y_cls[1:, 2:])
            mask = (ious < iou_thres)
            y_cls = y_cls[1:,:][mask]
            bb_size = y_cls.size(0)

    return y_res

def choose_color(idx, num=80):
    base = 80 // 3 + 1
    color = []
    while len(color) < 3:
        if idx > 0:
            color.append( min(int(idx * 255 / (base - 1)), 255) )
        else:
            color.append(0)
        idx -= base
    
    return tuple(color)

def draw_bboxes(img, bbs, cls_names, dim=416):
    h, w = img.shape[:2]
    x_scale = w / dim
    y_scale = h / dim

    cls_size = len(cls_names)
    for bb in bbs:
        cls = int(bb[0])
        color = choose_color(cls, cls_size)
        point1 = (int(bb[2]*x_scale), int(bb[3]*y_scale))
        point2 = (int(bb[4]*x_scale), int(bb[5]*y_scale))
        cv2.rectangle(img, point1, point2, color, 1)

        text = cls_names[cls] + ': %d%%' % int(bb[1]*100)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6 , 1)[0]
        cv2.rectangle(img, point1, (point1[0]+text_size[0]+2, point1[1]+text_size[1]+2), color, -1)
        cv2.putText(img, text, (point1[0]+1, point1[1]+text_size[1]+1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, [225,255,255], 1)

    return img

