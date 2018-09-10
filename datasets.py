import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from utils_train import *
import cv2

voc_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class PascalVosDataset(Dataset):
    def __init__(self, txtfile, transform=None):
        self.images = [x.strip() for x in open(txtfile).readlines() if len(x) > 0]
        self.labels = ['/'.join(x.split('/')[:-2]) + '/labels/' + x.split('/')[-1].split('.')[0] + '.txt' for x in self.images]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx]) # BGR to RGB
        label = [x.strip().split(' ') for x in open(self.labels[idx]).readlines() if len(x) > 0]
        label_cls = np.array([int(x[0]) for x in label], dtype=np.int32)
        label_bbx = np.array([[float(x) for x in lbl[1:]] for lbl in label])  # x1, y1, x2, y2

        sample = {'image': image, 'label': (label_cls, label_bbx)}
        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale:
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size [h, w]. If tuple, output is matched to output_size. If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        label_cls, label_bbx = label

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h < w:
                h_new, w_new = int(self.output_size * h / w), self.output_size
            else:
                h_new, w_new = self.output_size, int(self.output_size * w / h)
        else:
            assert len(output_size) == 2
            h_new, w_new = self.output_size

        image_new = cv2.resize(image, (w_new, h_new), cv2.INTER_LINEAR)
        label_bbx[:, ::2] *= w_new / w
        label_bbx[:, 1::2] *= h_new / h
        label_new = (label_cls, label_bbx)

        sample = {'image': image_new, 'label': label_new}

        return sample


class RandomHorizontalFlip:
    """Horizontally flip the image in a sample randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5.
    """

    def __init__(self, p=0.5):
        assert isinstance(p, (int, float))
        assert 0 <= p <= 1 
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        label_cls, label_bbx = label

        h, w = image.shape[:2]

        flip = False
        if np.random.rand() < self.p:
            flip = True
            image_new = cv2.flip(image, 1)
        else:
            image_new = image

        if flip:
            x1_new = w - 1 - label_bbx[:, 2]
            x2_new = w - 1 - label_bbx[:, 0]
            label_bbx[:, 0] = x1_new
            label_bbx[:, 2] = x2_new

        label_new = (label_cls, label_bbx)
        sample = {'image': image_new, 'label': label_new}

        return sample


class BottomRightPad:
    """Pad the image in a sample.

    Args:
        output_size (tuple or int): Desired output size [h, w]. If tuple, output is matched to output_size. If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
        fill (int): Pixel fill value. Default is 0. 
    """

    def __init__(self, output_size, fill=0):
        assert isinstance(output_size, (int, tuple))
        assert isinstance(fill, int)
        self.output_size = output_size
        self.fill = fill

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            h_new, w_new = self.output_size, self.output_size
        else:
            assert len(output_size) == 2
            h_new, w_new = self.output_size

        image_new = np.pad(image, ((0, h_new - h), (0, w_new - w), (0, 0)), 'constant', constant_values=self.fill)

        sample = {'image': image_new, 'label': label}

        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image_new = image[:,:,::-1].transpose((2,0,1)) / 255.0
        image_new = torch.from_numpy(image_new).float().unsqueeze(0)

        sample = {'image': image_new, 'label': label}

        return sample


if __name__ == '__main__':
    dtset = PascalVosDataset('/data/liuf/PascalVoc/voc_train.txt', transforms.Compose([Rescale(416), RandomHorizontalFlip(), BottomRightPad(416)]))

    sample = dtset[1]
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

    print(label)
    cv2.imwrite('test_dataset.jpg', img)

