import torch
from yolo_model.yolo_constraints import *
import cv2
from pascal_voc.voc_detection import decode_output_tensor

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

classid2color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def predict(input_img, predictor, nms=True):
    objs_list = list()

    img_shape = input_img.shape

    if img_shape[0] != INPUT_TENSOR_C:
        input_img = input_img.reshape([img_shape[2], img_shape[0], img_shape[1]])
        img_shape = input_img.shape
        if img_shape[0] != INPUT_TENSOR_C:
            print("Error! Cant fit image, because of wrong input shape")

    if img_shape[1] != INPUT_TENSOR_X or img_shape[2] != INPUT_TENSOR_Y:
        cv2.resize(input_img, dsize=(INPUT_TENSOR_X, INPUT_TENSOR_Y))

    input_tensor = torch.tensor(input_img, dtype=torch.float32)
    input_tensor = input_tensor.reshape(1, INPUT_TENSOR_C, INPUT_TENSOR_Y, INPUT_TENSOR_X)

    out_tensor = predictor.forward(input_tensor)

    return decode_output_tensor(out_tensor)

    # for cell_y in range(7):
    #     for cell_x in range(7):
    #         cell_vector = out_tensor[0, :, cell_y, cell_x]
    #         probs = cell_vector[10:]
    #         box1 = cell_vector[:5]
    #         box2 = cell_vector[5:10]
    #         max_id = torch.argmax(probs).detach()
    #         prob = probs[max_id]
    #
    #         label = voc2007_id2class_dict[max_id.item()]
    #         if prob > 0.1:
    #             box = BBox()
    #             best_box = box1 if box1[4] >= box2[4] else box2
    #             box.set_norm(cell_x, cell_y, best_box[0], best_box[1], best_box[2], best_box[3])
    #             objs_list.append({"class": label, "bbox": box.get_abs_coords(), "prob": best_box[4] * prob})
    # return objs_list


def show_prediction(inp_img, objects_list):
    img = inp_img
    objs = objects_list

    if type(img) != np.ndarray:
        img = img.numpy()
    if img.shape[2] != 3:  # cv2 format
        img = np.reshape(img, (img.shape[1], img.shape[2], img.shape[0])) # make it pil

    img = np.add(img, 0.5)
    img = np.multiply(img, 255.)

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(img)
    # Create a Rectangle patch
    for obj in objs:
        _class = obj["class"]
        _bbox = obj["bbox"]
        rect = patches.Rectangle((_bbox[0], _bbox[1]), _bbox[2], _bbox[3],
                                 linewidth=1, edgecolor=classid2color[obj["class_id"] % len(classid2color)],
                                 facecolor='none')
        ax.add_patch(rect)
        print("Object label: ", _class, " confidence: ", obj["conf"], " prob: ", obj["prob"])
    print("Objects cnt: ", len(objects_list))
    plt.show()
    pass
