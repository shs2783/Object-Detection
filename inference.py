import os
import cv2

import torch
from torch.utils.data import DataLoader
import albumentations as A

from selective_search import region_proposal
from models import RCNN
from utils import refine_bbox, non_max_suppression, show_image_labels

def inference(model,
              image,
              batch_size=16,
              max_proposals=2000,
              padding=16,
              transforms=None,
              apply_nms=True,
              nms_threshold=0.2,
              conf_threshold=0.5,
              device='cpu',
              ):

    model.eval()

    proposal_imgs, proposal_boxes = region_proposal(image, max_proposals, padding, transforms)
    proposal_imgs = torch.stack(proposal_imgs)
    proposal_boxes = torch.stack(proposal_boxes)

    proposal_imgs = DataLoader(proposal_imgs, batch_size=batch_size)
    proposal_bboxes = DataLoader(proposal_boxes, batch_size=batch_size)

    cnt = 0
    pred_bboxes = []
    for imgs, boxes in zip(proposal_imgs, proposal_bboxes):
        imgs = imgs.to(device)

        with torch.no_grad():
            class_pred, bbox_pred = model(imgs, bbox=True)
            class_pred = torch.softmax(class_pred, dim=-1)
            confidences, class_ids = class_pred.max(dim=-1)

        not_bg_preds = torch.where(class_ids > 0)
        class_ids = class_ids.cpu().tolist()
        confidences = confidences.cpu().tolist()

        for idx in not_bg_preds[0]:
            idx = idx.item()

            bbox = refine_bbox(boxes[idx], bbox_pred[idx])
            class_id = class_ids[idx]
            confidence = confidences[idx]

            pred_bboxes.append(bbox + [class_id, confidence])
        cnt += 1

    # apply non-max suppression to remove duplicate boxes
    if apply_nms:
        pred_bboxes = non_max_suppression(pred_bboxes, iou_threshold=nms_threshold, conf_threshold=conf_threshold)

    return pred_bboxes

if __name__ == '__main__':
    # config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {'image_size': 224, 'n_classes': 21, 'max_proposals': 2000, 'pad': 16}
    train_config = {'epochs': 5, 'batch_size': 128, 'lr': 0.001, 'lr_decay': 0.5, 'l2_reg': 1e-5, 'bbox_iou_threshold': 0.6}
    load_path = 'RCNN_checkpoint.pt'

    img_path = './dataset/pascalVOC/content/VOCtrain/VOCdevkit/VOC2012/JPEGImages'
    img_list = os.listdir(img_path)

    # transforms
    transforms = A.Compose([
        A.Resize(config['image_size'], config['image_size']),
        A.ToFloat(),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ], )

    # model
    model = RCNN(in_channels=3, num_classes=21, drop_out=0.2).to(device)
    check_point = torch.load(load_path)
    model.load_state_dict(check_point['model_state_dict'])

    for img_file in img_list:
        img_file_path = os.path.join(img_path, img_file)
        img = cv2.imread(img_file_path)

        preds = inference(model, img, transforms=transforms, device=device)
        print(preds)
        show_image_labels(img, preds)