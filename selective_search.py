import os
import gc
import cv2
import random
import pickle

from tqdm import tqdm
from pathlib import Path
from typing import Union

import torch
import numpy as np

from utils import load_image, save_image, crop_image, xyxy_to_xywh, intersection_over_union, parse_xml

labels = ['none', 'aeroplane', "bicycle", 'bird', "boat", "bottle", "bus", "car", "cat", "chair", 'cow', "diningtable", "dog", "horse", "motorbike", 'person', "pottedplant", 'sheep', "sofa", "train", "tvmonitor"]
converted_label = ['Background', 'Aeroplane', "Bicycle", 'Bird', "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", 'Cow', "Dining table", "Dog", "Horse", "Motorbike", 'Person', "Potted plant", 'Sheep', "Sofa", "Train", "TV/monitor"]

convert_labels_to_id = {}
for idx, label in enumerate(labels):
    convert_labels_to_id[label] = idx

def save_images_and_labels(img, labels, img_name, save_image_path, save_label_path, file_counter=0):
    image_file = f'{img_name[:-4]}_{file_counter}.jpg'
    label_file = f'{img_name[:-4]}_{file_counter}.pkl'

    save_image(save_image_path / image_file, img)
    with open(save_label_path / label_file, 'wb') as file:
        pickle.dump(labels, file)

def selective_search(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    return ss.process()

def region_proposal(image, max_proposals=2000, padding=16, transforms=None):
    roi_rects = selective_search(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    proposal_imgs = []
    proposal_boxes = []

    for (x1, y1, w, h) in roi_rects[:max_proposals]:
        x1, x2 = np.clip([x1 - padding, x1 + w + padding], 0, image.shape[1])
        y1, y2 = np.clip([y1 - padding, y1 + h + padding], 0, image.shape[0])
        roi = image[y1:y2, x1:x2, :]

        if transforms:
            roi = transforms(image=roi)
            roi = roi['image'].transpose(2, 0, 1)

        roi = torch.from_numpy(roi)
        proposal_imgs.append(roi)

        x = x1 + w//2
        y = y1 + h//2
        bbox = torch.LongTensor([x, y, w, h])
        proposal_boxes.append(bbox)

    return proposal_imgs, proposal_boxes

def generate_selective_search_roi(image_path: Union[Path, str], annotation_path: Union[Path, str]):
    print('[*] Generating Selective Search dataset for R-CNN.')

    if not isinstance(image_path, Path):
        image_path = Path(image_path)
    if not isinstance(annotation_path, Path):
        annotation_path = Path(annotation_path)

    image_list = os.listdir(image_path)
    annot_list = os.listdir(annotation_path)

    for image_name, annot_name in zip(image_list, annot_list):
        gc.collect()

        image = load_image(image_path / image_name)
        xml_path = annotation_path / annot_name
        gt_bboxes = parse_xml(xml_path)

        roi_rects = selective_search(image)[:2000]  # parse first 2000 boxes
        random.shuffle(roi_rects)
        yield image_name, image, gt_bboxes, roi_rects

def generate_bbox_dataset(image_path: Union[Path, str],
                          annotation_path: Union[Path, str],
                          fg_samples: int = 32,
                          bg_samples: int = 32,
                          IoU_threshold: float = 0.5,
                          padding: float = 16,
                          ):
    print('[*] Generating bbox dataset for R-CNN.')

    save_path = image_path.parent
    save_image_path = save_path / 'SelectiveSearchBboxImages'
    save_label_path = save_path / 'SelectiveSearchBboxLabels'

    os.makedirs(save_image_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)

    roi_generator = generate_selective_search_roi(image_path, annotation_path)
    pbar = tqdm(roi_generator, total=len(os.listdir(image_path)))
    for n, data in enumerate(roi_generator):
        img_name, image, gt_bboxes, roi_rects = data

        # define counters
        file_counter = 0
        obj_counter = 0
        bg_counter = 0

        # loop through all ss bounding box proposals
        for (x1, y1, w, h) in roi_rects:
            # apply padding
            x1, x2 = np.clip([x1 - padding, x1 + w + padding], 0, image.shape[1])
            y1, y2 = np.clip([y1 - padding, y1 + h + padding], 0, image.shape[0])
            bbox_est = [x1, y1, x2, y2]
            is_object = False  # define flag

            # check the proposal with every elements of the gt boxes
            for box_info in gt_bboxes:
                x1, y1, x2, y2, class_name = box_info
                class_id = convert_labels_to_id[class_name]
                class_name = converted_label[class_id]

                bbox_gt = [x1, y1, x2, y2]
                iou = intersection_over_union(bbox_gt, bbox_est, box_format='xyxy')

                if iou >= IoU_threshold:  # if object is close to g.t bbox
                    if obj_counter < fg_samples:
                        cropped_img = crop_image(image, bbox_est, box_format='xyxy')
                        est_bbox_xywh = xyxy_to_xywh(bbox_est)
                        gt_bbox_xywh = xyxy_to_xywh(bbox_gt)
                        labels = [class_id, class_name, est_bbox_xywh, gt_bbox_xywh]
                        save_images_and_labels(cropped_img, labels, img_name, save_image_path, save_label_path, file_counter)

                        obj_counter += 1
                        file_counter += 1

                    is_object = True
                    break

            # if the object is not close to any g.t bbox (= background)
            if not is_object and bg_counter < bg_samples:
                cropped_img = crop_image(image, bbox_est, box_format='xyxy')
                est_bbox_xywh = (1.0, 1.0, 1.0, 1.0)
                gt_bbox_xywh = (1.0, 1.0, 1.0, 1.0)
                labels = [0, 'Background', est_bbox_xywh, gt_bbox_xywh]
                save_images_and_labels(cropped_img, labels, img_name, save_image_path, save_label_path, file_counter)

                bg_counter += 1
                file_counter += 1

            # if the number of samples is enough, break the loop
            if obj_counter >= fg_samples and bg_counter >= bg_samples:
                break

        # update progress bar description
        pbar.set_description(f"image no.{n} - object: {obj_counter} background: {bg_counter}")
        pbar.update(1)

def generate_classifier_dataset(image_path: Union[Path, str],
                                annotation_path: Union[Path, str],
                                bg_samples: int = 5,
                                IoU_threshold: float = 0.3,
                                padding: float = 16,
                                ):
    print('[*] Generating classifier dataset for R-CNN.')

    save_path = image_path.parent
    save_image_path = save_path / 'SelectiveSearchClassImages'
    save_label_path = save_path / 'SelectiveSearchClassLabels'

    os.makedirs(save_image_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)

    roi_generator = generate_selective_search_roi(image_path, annotation_path)
    pbar = tqdm(roi_generator, total=len(os.listdir(image_path)))
    for n, data in enumerate(roi_generator):
        img_name, image, gt_bboxes, roi_rects = data

        # define counters
        file_counter = 0
        obj_counter = 0
        bg_counter = 0

        # save all gt bounding boxes
        for box_info in gt_bboxes:
            x1, y1, x2, y2, class_name = box_info
            class_id = convert_labels_to_id[class_name]
            class_name = converted_label[class_id]

            bbox_gt = [x1, y1, x2, y2]
            cropped_img = crop_image(image, bbox_gt, box_format='xyxy')
            est_bbox_xywh = xyxy_to_xywh(bbox_gt)
            gt_bbox_xywh = xyxy_to_xywh(bbox_gt)
            labels = [class_id, class_name, est_bbox_xywh, gt_bbox_xywh]
            save_images_and_labels(cropped_img, labels, img_name, save_image_path, save_label_path, file_counter)

            obj_counter += 1
            file_counter += 1

        # loop through all ss bounding box proposals
        for (x1, y1, w, h) in roi_rects:
            # apply padding
            x1, x2 = np.clip([x1 - padding, x1 + w + padding], 0, image.shape[1])
            y1, y2 = np.clip([y1 - padding, y1 + h + padding], 0, image.shape[0])
            bbox_est = [x1, y1, x2, y2]
            is_object = False  # define flag

            # check the proposal with every elements of the gt boxes
            for box_info in gt_bboxes:
                x1, y1, x2, y2, class_name = box_info
                class_id = convert_labels_to_id[class_name]
                class_name = converted_label[class_id]

                bbox_gt = [x1, y1, x2, y2]
                iou = intersection_over_union(bbox_gt, bbox_est, box_format='xyxy')

                if iou >= IoU_threshold:  # if object is close to g.t bbox
                    is_object = True
                    break

            # if the object is not close to any g.t bbox (= background)
            if not is_object and bg_counter < bg_samples:
                cropped_img = crop_image(image, bbox_est, box_format='xyxy')
                est_bbox_xywh = (1.0, 1.0, 1.0, 1.0)
                gt_bbox_xywh = (1.0, 1.0, 1.0, 1.0)
                labels = [0, 'Background', est_bbox_xywh, gt_bbox_xywh]
                save_images_and_labels(cropped_img, labels, img_name, save_image_path, save_label_path, file_counter)

                bg_counter += 1
                file_counter += 1

            # if the number of samples is enough, break the loop
            if bg_counter >= bg_samples:
                break

        # update progress bar description
        pbar.set_description(f"Data size - image no.{n} object: {obj_counter} background: {bg_counter}")
        pbar.update(1)

if __name__ == '__main__':
    image_path = Path('dataset/pascalVOC/content/VOCtrain/VOCdevkit/VOC2012/JPEGImages/')
    annotation_path = Path('dataset/pascalVOC/content/VOCtrain/VOCdevkit/VOC2012/Annotations/')

    generate_bbox_dataset(image_path, annotation_path)
    generate_classifier_dataset(image_path, annotation_path)