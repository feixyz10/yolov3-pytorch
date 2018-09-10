import torch 
import torch.nn.functional as F
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, anchors, grid_sizes=[13, 26, 52], inp_dim=416, num_classes=80, lamb=1):
        super(__class__, self).__init__()

        self.anchors = anchors
        self.grid_sizes = grid_sizes
        self.inp_dim = inp_dim
        self.num_classes = num_classes
        self.lamb = lamb

        self.huberLoss = nn.SmoothL1Loss()
        self.bceLogitLoss = nn.BCEWithLogitsLoss()

    def forward(self, x, label, pos, neg):
        label_cls, label_bbx = label
        num_classes = self.num_classes

        grid_sizes = self.grid_sizes
        anchors = self.anchors

        loss_obj = 0
        loss_bbx = 0
        
        neg = [torch.Tensor([[(i % (grid_sizes[j] * 3)) % 3, i // (grid_sizes[j] * 3), (i % (grid_sizes[j] * 3)) // 3] for i, n in enumerate(ng) if not n]).type(torch.int64) for j, ng in enumerate(neg)]

        for i, ng in enumerate(neg):
            if ng.numel() != 0:
                y_ = x[i][0, ng[:,0]*num_classes+4, ng[:,1], ng[:,2]]
                loss_obj += self.bceLogitLoss(y_, torch.zeros_like(y_, device=y_.device))
        print(loss_obj)

        for j, ps in enumerate(pos):
            i, anc_idx, _, x_idx, y_idx = ps
            stride = self.inp_dim // grid_sizes[j]
            anc = anchors[j][i]

            y_obj = x[i][0, anc_idx*num_classes+4, y_idx, x_idx]
            loss_obj += self.bceLogitLoss(y_obj, torch.ones_like(y_obj, device=y_obj.device))
            print(loss_obj)
            
            y_cls = x[i][0, anc_idx*num_classes+5:anc_idx*num_classes+85, y_idx, x_idx]
            t_cls = torch.zeros_like(y_cls, device=y_cls.device)
            t_cls[label_cls[j]] = 1.0
            loss_obj += self.bceLogitLoss(y_cls, t_cls)
            print(loss_obj)

            y_bbx = x[i][0, anc_idx*num_classes:anc_idx*num_classes+4, y_idx, x_idx]
            t_bbx = torch.from_numpy(label_bbx[j]).float()
            x_c = (t_bbx[0] + t_bbx[2]) / 2
            y_c = (t_bbx[1] + t_bbx[3]) / 2
            w = t_bbx[2] - t_bbx[0]
            h = t_bbx[3] - t_bbx[1]
            t_bbx[0] = (x_c - x_idx * stride) / anc[0]
            t_bbx[1] = (y_c - y_idx * stride) / anc[1]
            t_bbx[2] = torch.log(w / anc[0])
            t_bbx[3] = torch.log(h / anc[1])

            loss_bbx += self.huberLoss(y_bbx, t_bbx)
            print(loss_bbx)

        return loss_obj + self.lamb * loss_bbx


if __name__ == '__main__':
    from utils_train import *
    from models import *
    from datasets import *
    anchors = [ [[116,90], [156,198], [373,326]],  [[30,61], [62,45], [59,119]],  [[10,13], [16,30], [33,23]] ]  # yolov3

    yolo3 = Yolo3()
    dtset = PascalVosDataset('/data/liuf/PascalVoc/voc_train.txt', transforms.Compose([Rescale(416), RandomHorizontalFlip(), BottomRightPad(416), ToTensor()]))
    sample = dtset[1]
    img, label = sample['image'], sample['label']
    label_cls, label_bbx = label
    pos, neg = pos_neg_anchors(label, anchors)

    yolo3.load_state_dict(torch.load('yolov3.pth'))
    yolo3.train(False)
    y = yolo3(img)
    loss = YoloLoss(anchors)

    print(loss(y, label, pos, neg))
