import os
import cv2
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset

from typing import Optional
from selective_search import generate_bbox_dataset, generate_classifier_dataset

def load_image(img_path):
    if isinstance(img_path, Path):
        img_path = str(img_path)

    img = cv2.imread(img_path)
    return img

class ImageDataset(Dataset):
    def __init__(self,
                 img_path: str,
                 label_path: str,
                 transforms: Optional = None,
                 data_type: str = 'bbox',
                 ):

        self.img_path = Path(img_path)
        self.label_path = Path(label_path)

        if data_type == 'bbox':
            if not os.path.exists(self.img_path.parent / 'SelectiveSearchBboxImages'):
                generate_bbox_dataset(self.img_path, self.label_path)

            self.img_path = self.img_path.parent / 'SelectiveSearchBboxImages'
            self.label_path = self.label_path.parent / 'SelectiveSearchBboxLabels'

        elif data_type == 'class':
            if not os.path.exists(self.img_path.parent / 'SelectiveSearchClassImages'):
                generate_classifier_dataset(self.img_path, self.label_path)

            self.img_path = self.img_path.parent / 'SelectiveSearchClassImages'
            self.label_path = self.label_path.parent / 'SelectiveSearchClassLabels'

        self.img_list = os.listdir(self.img_path)
        self.label_list = os.listdir(self.label_path)

        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = load_image(self.img_path / self.img_list[idx])

        with open(self.label_path / self.label_list[idx], 'rb') as f:
            class_id, class_name, est_bbox, gt_bbox = pickle.load(f)

        if self.transforms:
            image = self.transforms(image=image)
            image = image['image']
            image = image.transpose(2, 0, 1)

        return {
            'image': image,
            'class_name': class_name,
            'class_id': torch.tensor(class_id),
            'est_bbox': torch.tensor(est_bbox),
            'gt_bbox': torch.tensor(gt_bbox),
        }

if __name__ == '__main__':
    img_path = './dataset/pascalVOC/content/VOCtrain/VOCdevkit/VOC2012/JPEGImages'
    annotation_path = './dataset/pascalVOC/content/VOCtrain/VOCdevkit/VOC2012/Annotations'

    image_dataset = ImageDataset(img_path, annotation_path)
    print(len(image_dataset))