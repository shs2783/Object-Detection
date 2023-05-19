import cv2
import math
import logging

import numpy as np
from pathlib import Path

import torch
import xml.etree.ElementTree as ET

def get_logger(name,
               format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
               date_format='%Y-%m-%d %H:%M:%S',
               file=False):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(name) if file else logging.StreamHandler()
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def show_params(model):
    print("=" * 40, "Model Parameters", "=" * 40)
    num_params = 0
    for module_name, m in model.named_modules():
        if module_name == '':
            for name, params in m.named_parameters():
                print(name, params.size(), end=' ')

                i = params.numel()
                if 'weight' in name:
                    print('>'.rjust(20, '-'), i)
                else:
                    print()
                    
                num_params += i
    print('[*] Parameter Size: {}'.format(num_params))
    print("=" * 98)

    return num_params

def initialize_params(model, nonlinearity='relu', weight_norm=False):
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity=nonlinearity)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias.data)
            if weight_norm:
                torch.nn.utils.weight_norm(module)

        elif isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)):
            torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity=nonlinearity)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias.data)
            if weight_norm:
                torch.nn.utils.weight_norm(module)

        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data, nonlinearity=nonlinearity)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias.data)
            if weight_norm:
                torch.nn.utils.weight_norm(module)

        elif isinstance(module, torch.nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)

        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            torch.nn.init.constant_(module.weight.data, 1)
            torch.nn.init.constant_(module.bias.data, 0)

def train_test_split(len_dataset, train_ratio=0.8, shuffle=True):
    indices = list(range(len_dataset))

    if shuffle:
        np.random.shuffle(indices)

    split = int(train_ratio * len_dataset)
    train_idx, test_idx = indices[:split], indices[split:]

    return train_idx, test_idx

def parse_xml(xml_path):
    box_info = []

    with open(xml_path, 'r') as file:
        tree = ET.parse(file)
        root = tree.getroot()

    objects = root.findall("object")
    for _object in objects:
        class_name = _object.find('name').text.lower()
        bbox = _object.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        box_info.append([xmin, ymin, xmax, ymax, class_name])

    return box_info

def load_image(img_path):
    if isinstance(img_path, Path):
        img_path = str(img_path)

    img = cv2.imread(img_path)
    return img


def save_image(save_path, img):
    if isinstance(save_path, Path):
        save_path = str(save_path)

    cv2.imwrite(save_path, img)

def crop_image(image, bbox, box_format='xyxy'):
    if box_format == 'xyxy':
        x1, y1, x2, y2 = bbox
    elif box_format == 'xywh':
        x, y, w, h = bbox
        x1 = x
        y1 = y
        x2 = x1 + w
        y2 = y1 + h

    return image[y1:y2, x1:x2, :]

def xyxy_to_xywh(box, absolute=True):
    ''' [x1, y1, x2, y2] -> [x_center, y_center, width, height] '''
    x1, y1, x2, y2 = box[:4]

    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2

    if absolute:
        width = int(width)
        height = int(height)
        x_center = int(x_center)
        y_center = int(y_center)

    box[:4] = [x_center, y_center, width, height]
    return box

def xywh_to_xyxy(box, absolute=True):
    ''' [x_center, y_center, width, height] -> [x1, y1, x2, y2] '''
    x_center, y_center, width, height = box[:4]

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    if absolute:
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

    box[:4] = x1, y1, x2, y2
    return box

def intersection_over_union(pred_box, target_box, box_format='xywh'):
    """
    Calculates intersection over union
    Parameters:
        pred_box (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        target_box (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if isinstance(pred_box, np.ndarray):
        pred_box = torch.from_numpy(pred_box)
        target_box = torch.from_numpy(target_box)
    elif isinstance(pred_box, list):
        pred_box = torch.FloatTensor(pred_box)
        target_box = torch.FloatTensor(target_box)

    if box_format == "xywh":
        box1_x1 = pred_box[..., 0] - pred_box[..., 2] / 2
        box1_y1 = pred_box[..., 1] - pred_box[..., 3] / 2
        box1_x2 = pred_box[..., 0] + pred_box[..., 2] / 2
        box1_y2 = pred_box[..., 1] + pred_box[..., 3] / 2
        box2_x1 = target_box[..., 0] - target_box[..., 2] / 2
        box2_y1 = target_box[..., 1] - target_box[..., 3] / 2
        box2_x2 = target_box[..., 0] + target_box[..., 2] / 2
        box2_y2 = target_box[..., 1] + target_box[..., 3] / 2

    elif box_format == "xyxy":
        box1_x1 = pred_box[..., 0]
        box1_y1 = pred_box[..., 1]
        box1_x2 = pred_box[..., 2]
        box1_y2 = pred_box[..., 3]
        box2_x1 = target_box[..., 0]
        box2_y1 = target_box[..., 1]
        box2_x2 = target_box[..., 2]
        box2_y2 = target_box[..., 3]

    box1_width = box1_x2 - box1_x1
    box1_height = box1_y2 - box1_y1
    box1_area = torch.abs(box1_width * box1_height)

    box2_width = box2_x2 - box2_x1
    box2_height = box2_y2 - box2_y1
    box2_area = torch.abs(box2_width * box2_height)

    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    inter_width = (inter_x2 - inter_x1).clamp(0)
    inter_height = (inter_y2 - inter_y1).clamp(0)
    intersection = inter_width * inter_height

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, conf_threshold, box_format='xywh'):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list, tensor or ndarray): array of lists containing all bboxes with each bboxes
        specified as [x, y, w, h, confidence_score, class_pred]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "xywh" or "xyxy"

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    if len(bboxes) == 0:
        return []

    if isinstance(bboxes, list):
        bboxes = torch.FloatTensor(bboxes)
    elif isinstance(bboxes, np.ndarray):
        bboxes = torch.from_numpy(bboxes)

    bboxes = bboxes[bboxes[..., 4] >= conf_threshold]  # (N, 6)  # 6 = (x, y, w, h, confidence_score, class_pred)
    bboxes = bboxes[bboxes[..., 4].argsort()]

    bboxes_after_nms = []
    while len(bboxes):
        chosen_box = bboxes[-1]
        bboxes = bboxes[:-1]

        ious = intersection_over_union(bboxes[..., :4], chosen_box[:4], box_format)  # (N, 1)  ## 1 = iou
        bboxes = bboxes[ious <= iou_threshold]

        bboxes_after_nms.append(chosen_box.tolist())

    return bboxes_after_nms

# def nms(P, iou_threshold=0.5):
#     # P: list of dicts {'bbox':(x1,y1,x2,y2), 'conf':float, 'class':int}
#     conf_list = np.array([x['conf'] for x in P])
#     conf_order = (-conf_list).argsort()  # apply minus to reverse order !!
#     isremoved = [False for _ in range(len(P))]
#     keep = []
#
#     for idx in range(len(P)):
#         to_keep = conf_order[idx]
#         if isremoved[to_keep]:
#             continue
#
#         # append to keep list
#         keep.append(P[to_keep])
#         isremoved[to_keep] = True
#         # remove overlapping bboxes
#         for order in range(idx + 1, len(P)):
#             bbox_idx = conf_order[order]
#             if isremoved[bbox_idx] == False:  # if not removed yet
#                 # check overlapping
#                 iou = calculate_iou(P[to_keep]['bbox'], P[bbox_idx]['bbox'])
#                 if iou > iou_threshold:
#                     isremoved[bbox_idx] = True
#     return keep

def mean_average_precision(pred_boxes, gt_boxes, iou_threshold=0.5, box_format='xywh', num_classes=20):
    '''
    pred_boxes (list): list of lists containing all bboxes with each bboxes
    specified as [train_idx, x1, y1, x2, y2, prob_score, class_prediction]
    '''

    if isinstance(pred_boxes, list):
        pred_boxes = torch.FloatTensor(pred_boxes)
        gt_boxes = torch.FloatTensor(gt_boxes)
    elif isinstance(pred_boxes, np.ndarray):
        pred_boxes = torch.from_numpy(pred_boxes)
        gt_boxes = torch.from_numpy(gt_boxes)

    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = pred_boxes[pred_boxes[..., -1] == c]
        ground_truths = gt_boxes[gt_boxes[..., -1] == c]

        # If none exists for this class then we can safely skip
        total_true_bboxes = len(ground_truths)
        if total_true_bboxes == 0:
            continue

        amount_bboxes = dict(Counter(ground_truths[..., 0].tolist()))  # count train_idx
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 5
        detections = detections[detections[..., 5].argsort()]
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = ground_truths[ground_truths[..., 0] == detection[0]]
            if len(ground_truth_img):
                ious = intersection_over_union(detection[1:], ground_truth_img[..., 1:], box_format)
                best_iou = torch.max(ious)
                best_iou_idx = torch.argmax(ious)

                batch_idx = detection[0].item()
                if best_iou > iou_threshold:
                    # only detect ground truth detection once
                    if amount_bboxes[batch_idx][best_iou_idx] == 0:
                        # true positive and add this bounding box to seen
                        amount_bboxes[batch_idx][best_iou_idx] = 1
                        TP[detection_idx] = 1
                    else:
                        FP[detection_idx] = 1

                # if IOU is lower then the detection is a false positive
                else:
                    FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)  # ex) [1, 1, 0, 0, 1, 0, 1] -> [1, 2, 2, 2, 3, 3, 4]
        FP_cumsum = torch.cumsum(FP, dim=0)  # ex) [0, 0, 1, 1, 0, 1, 0] -> [0, 0, 1, 2, 2, 3, 3]

        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)

        # torch.trapz for numerical integration
        recalls = torch.cat((torch.tensor([0]), recalls))
        precisions = torch.cat((torch.tensor([1]), precisions))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def refine_bbox(proposal_bbox, pred_bbox):
    if isinstance(proposal_bbox, torch.Tensor):
        proposal_bbox = proposal_bbox.tolist()
    if isinstance(pred_bbox, torch.Tensor):
        pred_bbox = pred_bbox.tolist()

    px, py, pw, ph = proposal_bbox
    tx, ty, tw, th = pred_bbox

    new_x = px + (pw * tx)
    new_y = py + (ph * ty)
    new_w = pw * math.exp(tw)
    new_h = ph * math.exp(th)

    x1 = new_x - new_w/2
    y1 = new_y - new_h/2
    x2 = new_x + new_w/2
    y2 = new_y + new_h/2

    return [x1, y1, x2, y2]

def show_image_labels(image, bboxes):
    for bbox in bboxes:
        x1, y1, x2, y2, class_id, confidence_score = bbox
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{class_id} {confidence_score:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('image', image)
    cv2.waitKey(0)