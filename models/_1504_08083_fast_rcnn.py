'https://github.com/gary1346aa/Fast-RCNN-Object-Detection-Pytorch'
'https://arxiv.org/pdf/1504.08083.pdf'

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms
from torchvision.models import vgg16_bn

from utils import convert_to_relative_coordinate

def selective_search(image):
    # return region proposals of selective search over an image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return ss.process()



class ROIPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(output_size)
        self.output_size = output_size

    def roi_projection(self, image, roi):
        ...

    def forward(self, images, rois, roi_idx):
        '''
        roi 좌표 = 상대좌표 (0~1)
        rois = [
                 [[x1, y1, x2, y2], img1 rois
                  [x1, y1, x2, y2]],

                 [[x1, y1, x2, y2], img2 rois
                 [x1, y1, x2, y2]],

                 [[x1, y1, x2, y2], img3 rois
                 [x1, y1, x2, y2]]
               ]
        rois.shape = (n_img, n_roi, 4)
        roi_idx = [roi_idx1, roi_idx2, roi_idx3, ...]
        roi_idx.shape = (n_roi, )
        '''

        n, c, h, w = images.size()
        x1 = rois[:, 0]
        y1 = rois[:, 1]
        x2 = rois[:, 2]
        y2 = rois[:, 3]

        x1 = np.floor(x1 * w).astype(int)
        x2 = np.ceil(x2 * w).astype(int)
        y1 = np.floor(y1 * h).astype(int)
        y2 = np.ceil(y2 * h).astype(int)

        result = []
        for i in range(len(rois)):
            img = images[roi_idx[i]].unsqueeze(0)
            img = img[:, :, y1[i]:y2[i], x1[i]:x2[i]]
            img = self.max_pool(img)
            result.append(img)

        result = torch.cat(result, dim=0)
        return result

class FastRCNN(nn.Module):
    def __init__(self, num_classes=20, max_proposal=2000):
        super().__init__()
        self.num_classes = num_classes
        self.max_proposal = max_proposal

        vgg_net = vgg16_bn(pretrained=True)
        self.conv = nn.Sequential(*list(vgg_net.features.children())[:-1])
        self.roi_pool = ROIPool(output_size=(7, 7))
        self.feature = nn.Sequential(*list(vgg_net.classifier.children())[:-1])

        self.cls_score = nn.Linear(feature_dim, num_classes + 1)
        self.bbox = nn.Linear(feature_dim, 4 * (num_classes + 1))

    def forward(self, x, rois, roi_idx):
        x = self.conv(x)
        x = self.roi_pool(x, rois, roi_idx)
        x = x.detach()
        x = x.view(x.size(0), -1)
        feat = self.feature(x)

        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat).view(-1, self.num_classes + 1, 4)
        return cls_score, bbox

    def region_proposal(self, image, rgb=False):
        # convert rgb to bgr for selective search
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if rgb else image

        # perform selective search to find region proposals
        rects = selective_search(image)

        # rect = (x1, y1, w, h)
        # box = (x1, y1, x2, y2)
        boxes = []
        for rect in rects[:self.max_proposals]:
            x1, y1, w, h = convert_to_relative_coordinate(image, rect)
            boxes.append([x1, y1, x1 + w, y1 + h])

        return boxes