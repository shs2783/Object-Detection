import torch

import albumentations as A
from image_dataset import ImageDataset
from torch.utils.data import DataLoader

from models import RCNN
from trainer import Trainer

if __name__ == '__main__':
    # config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {'image_size': 224, 'n_classes': 21, 'max_proposals': 2000, 'pad': 16}
    train_config = {'epochs': 5, 'batch_size': 64, 'lr': 0.001, 'lr_decay': 0.5, 'l2_reg': 1e-5, 'bbox_iou_threshold': 0.6}
    load_path = 'RCNN_checkpoint.pt'
    load_path = None

    # transforms
    transforms = A.Compose([
        A.Resize(config['image_size'], config['image_size']),
        A.ToFloat(),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ], )
    # bbox_params = A.BboxParams(format='pascal_voc', min_area=1024, min_visibility=0.1)

    # model
    model = RCNN(in_channels=3, num_classes=21, drop_out=0.2).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['l2_reg'])

    # scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_config['lr_decay'])

    # trainer
    trainer = Trainer(model, optimizer, scheduler, load_path, device=device)

    # dataset path
    img_path = './dataset/pascalVOC/content/VOCtrain/VOCdevkit/VOC2012/JPEGImages'
    annotation_path = './dataset/pascalVOC/content/VOCtrain/VOCdevkit/VOC2012/Annotations'

    # train bbox predictor
    voc_dataset = ImageDataset(img_path, annotation_path, transforms=transforms, data_type='bbox')
    voc_dataset = torch.utils.data.Subset(voc_dataset, indices=range(0, 20000))
    train_data_loader = DataLoader(voc_dataset, batch_size=train_config['batch_size'], num_workers=0, pin_memory=True, shuffle=True, drop_last=True)

    trainer.train(train_data_loader, train_bbox=True)

    # train classifier
    voc_dataset = ImageDataset(img_path, annotation_path, transforms=transforms, data_type='class')
    train_data_loader = DataLoader(voc_dataset, batch_size=train_config['batch_size'], num_workers=0, pin_memory=True, shuffle=True, drop_last=True)

    trainer.current_epoch = 0
    trainer.optimizer = torch.optim.Adam(model.classifier.parameters(), lr=train_config['lr'], weight_decay=train_config['l2_reg'])
    trainer.scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer.optimizer, gamma=train_config['lr_decay'])
    trainer.train(train_data_loader, train_bbox=False)