import numpy as np
from utils.bbox import BBox
import torch
from yolo_model.yolo_constraints import *
from torch.nn.functional import softmax, mse_loss
from torch import sigmoid
from utils.img_proc import *


def iou_tensors(tensor_rect_a, tensor_rect_b, cell_x, cell_y):
    a, b = BBox(), BBox()
    a.set_norm(cell_x, cell_y, tensor_rect_a[0], tensor_rect_a[1], tensor_rect_a[2], tensor_rect_a[3])
    b.set_norm(cell_x, cell_y, tensor_rect_b[0], tensor_rect_b[1], tensor_rect_b[2], tensor_rect_b[3])
    iou = a.iou(b)
    return iou


def compute_iou(pred, target):
    # pred and target shape = (-1, BBOX_CNT*4)
    # print(pred.shape, target.shape)

    # transform norm coords (x', y', w', h') into common scale and commin coord system
    # center - zero of current cell
    # size - norm to cell size [0...7.0]
    w1, h1 = pred[:, 2] * OUTPUT_TENSOR_X, pred[:, 3] * OUTPUT_TENSOR_Y
    w2, h2 = target[:, 2] * OUTPUT_TENSOR_X, target[:, 3] * OUTPUT_TENSOR_Y

    # now - recalculate x',y' to x0, y0 (relative )
    x1, y1 = pred[:, 0] - 0.5 * w1, pred[:, 1] - 0.5 * h1
    x2, y2 = target[:, 0] - 0.5 * w2, target[:, 1] - 0.5 * h2

    intersection_x0 = torch.max(x1, x2)
    intersection_x1 = torch.min(x1+w1, x2+w2)
    x_invalid = intersection_x1 <= intersection_x0
    # if intersection_x1 <= intersection_x0:
    #     return 0.
    intersection_y0 = torch.max(y1, y2)
    intersection_y1 = torch.min(y1+h1, y2+h2)
    y_invalid = intersection_y1 <= intersection_y0
    # if intersection_y1 <= intersection_y0:
    #     return 0.
    intersection_area = (intersection_x1 - intersection_x0) * (intersection_y1 - intersection_y0)
    intersection_area[x_invalid] = 0.0
    intersection_area[y_invalid] = 0.0
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou


def yolo_v1_loss(prediction_tensor, target_tensor, lambda_coord=5., lambda_noobj=0.5):
    # tensors shape - [-1, 30, 7, 7]
    center_pos_loss = 0.
    size_loss = 0.
    obj_conf_loss = 0.
    noobj_conf_loss = 0.
    classif_loss = 0.

    # prediction_tensor = prediction_tensor[0, :, :, :]

    # reshape tensors to more convinient form: spatial -> channels
    # prediction_tensor = prediction_tensor.view(-1, OUTPUT_TENSOR_Y, OUTPUT_TENSOR_X, OUTPUT_TENSOR_C)
    # target_tensor = target_tensor.view(-1, OUTPUT_TENSOR_Y, OUTPUT_TENSOR_X, OUTPUT_TENSOR_C)
    prediction_tensor = tensor_cyx2yxc(prediction_tensor)
    target_tensor = tensor_cyx2yxc(target_tensor)

    prediction_bboxes_conf = prediction_tensor[:, :, :, :CLASSES_DISTR_OFFSET]
    prediction_probs = prediction_tensor[:, :, :, CLASSES_DISTR_OFFSET:]
    target_bboxes_conf = target_tensor[:, :, :, :CLASSES_DISTR_OFFSET]
    target_probs = target_tensor[:, :, :, CLASSES_DISTR_OFFSET:]

    # apply scaling sigmoids for x,y,w,h,conf
    prediction_bboxes_conf = sigmoid(prediction_bboxes_conf)  # element wise
    # apply scaling softmax for classes
    # prediction_probs = softmax(prediction_probs, dim=3)  # dim=2)
    prediction_probs = sigmoid(prediction_probs)  # replace softmax to sigmoid (as for regression part)

    # boolean masks for ground truth objects position
    obj_pos = target_bboxes_conf[:, :, :, BBOX_C_POS] > 0.
    noobj_pos = ~obj_pos

    # XYWH pars ind
    # bboxes_indices = []
    # for i in range(BBOX_CNT):
    #     bboxes_indices.append(BBOX_X_POS + BBOX_STRUCT_LEN * i)
    #     bboxes_indices.append(BBOX_Y_POS + BBOX_STRUCT_LEN * i)
    #     bboxes_indices.append(BBOX_W_POS + BBOX_STRUCT_LEN * i)
    #     bboxes_indices.append(BBOX_H_POS + BBOX_STRUCT_LEN * i)
    # # C pars ind
    # conf_indices = []
    # for i in range(BBOX_CNT):
    #     conf_indices.append(BBOX_C_POS + BBOX_STRUCT_LEN * i)

    bboxes1_indices = [BBOX_X_POS, BBOX_Y_POS, BBOX_W_POS, BBOX_H_POS]
    bboxes2_indices = [BBOX_X_POS + BBOX_STRUCT_LEN, BBOX_Y_POS + BBOX_STRUCT_LEN, BBOX_W_POS + BBOX_STRUCT_LEN,
                       BBOX_H_POS + BBOX_STRUCT_LEN]
    conf1_indices = [BBOX_C_POS]
    conf2_indices = [BBOX_C_POS + BBOX_STRUCT_LEN]

    # split tensors by functions
    prediction_bboxes1 = prediction_bboxes_conf[:, :, :, bboxes1_indices]
    prediction_bboxes2 = prediction_bboxes_conf[:, :, :, bboxes2_indices]
    prediction_conf1 = prediction_bboxes_conf[:, :, :, conf1_indices]
    prediction_conf2 = prediction_bboxes_conf[:, :, :, conf2_indices]
    target_conf1 = target_bboxes_conf[:, :, :, conf1_indices]
    target_conf2 = target_bboxes_conf[:, :, :, conf2_indices]
    target_bboxes = target_bboxes_conf[:, :, :, bboxes1_indices]

    positive_pred_bboxes1 = prediction_bboxes1[obj_pos]
    positive_pred_bboxes2 = prediction_bboxes2[obj_pos]
    positive_target_bboxes = target_bboxes[obj_pos]

    positive_target_conf1 = compute_iou(positive_pred_bboxes1, positive_target_bboxes)
    positive_target_conf2 = compute_iou(positive_pred_bboxes2, positive_target_bboxes)

    positive_pred_conf1 = prediction_conf1[obj_pos].squeeze()
    positive_pred_conf2 = prediction_conf2[obj_pos].squeeze()

    negative_pred_conf1 = prediction_conf1[noobj_pos]
    negative_pred_conf2 = prediction_conf2[noobj_pos]

    negative_target_conf1 = target_conf1[noobj_pos]
    negative_target_conf2 = target_conf2[noobj_pos]

    positive_prediction_probs = prediction_probs[obj_pos]
    positive_target_probs = target_probs[obj_pos]

    # compute MSE for all tensors
    bboxes_mse = mse_loss(positive_pred_bboxes1, positive_target_bboxes) + \
                 mse_loss(positive_pred_bboxes2, positive_target_bboxes)
    pos_conf_mse = mse_loss(positive_pred_conf1, positive_target_conf1) + \
                   mse_loss(positive_pred_conf2, positive_target_conf2)
    neg_conf_mse = mse_loss(negative_pred_conf1, negative_target_conf1) + \
                   mse_loss(negative_pred_conf2, negative_target_conf2)
    probs_mse = mse_loss(positive_prediction_probs, positive_target_probs)
    # bboxes_mse = mse_loss(positive_prediction_bboxes, positive_target_bboxes)
    # pos_conf_mse = mse_loss(positive_prediction_confs, positive_target_confs)
    # neg_conf_mse = mse_loss(negative_prediction_confs, negative_target_confs)
    # probs_mse = mse_loss(positive_prediction_probs, positive_target_probs)

    return lambda_coord * bboxes_mse, pos_conf_mse, lambda_noobj * neg_conf_mse, probs_mse
