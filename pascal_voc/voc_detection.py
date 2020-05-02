import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, transform
import cv2
from utils.img_proc import *
import copy
from pascal_voc.voc_index import VocIndex
from pascal_voc.voc_constraints import *
from yolo_model.yolo_constraints import *
from utils.bbox import BBox


def reload_label_map():
    ini_path = "label_map_path.txt"
    class2id_dict = dict()
    id2class_dict = dict()
    if isfile(ini_path):
        with open(ini_path, "r") as f:
            map_file = f.readline().replace("\n", "")
            if isfile(map_file):
                with open(map_file, "r") as mf:
                    pair = mf.readline().replace("\n", "")
                    while pair:
                        pair = pair.split()
                        class2id_dict[pair[1]] = int(pair[0])
                        id2class_dict[int(pair[0])] = pair[1]
                        pair = mf.readline()
                    print("Class 2 id / id 2 class dictionaries updates")
    return class2id_dict, id2class_dict


class2id, id2class = reload_label_map()


class VocDetection(Dataset):
    """Pascal VOC 2007 dataset for object detection"""

    def __init__(self, root_dir, subset="trainval", size_limit=None, target_shape=None, transform=None):
        """
        Args:
            root_dir (string): Path to the Pascal VOC dataset 2007 with annotations.
            subset (string): 'trainval' or 'test' subset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.subset = subset
        self.tgt_shape = target_shape

        self.voc_index = VocIndex(self.root_dir, size_limit=size_limit)

        self.index = None
        if self.subset == "test":
            self.index = self.voc_index.get_test()
        else:
            self.index = self.voc_index.get_trainval()

        self.transform = transform
        pass

    def __len__(self):
        return self.index.get_size()

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        if isinstance(idx, slice):
            # Get the start, stop, and step from the slice
            return [self[i] for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, int):
            # if idx < 0:
            #     idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError("The index (%d) is out of range.", idx)
            return self.get_sample_by_id(idx)  # Get the data from elsewhere
        else:
            raise TypeError("Invalid argument type.")

    def get_sample_by_id(self, idx):
        img_desc = self.index.get_img_description(idx)
        img_name = img_desc.get_path()
        image = cv2.imread(img_name)  # read yxc opencv image with bgr color sequence

        ground_truth_boxes = img_desc.get_objects()

        if self.tgt_shape is not None:
            kx = self.tgt_shape[0] / image.shape[0]
            ky = self.tgt_shape[1] / image.shape[1]

            # image = transform.resize(image, (self.tgt_shape[0], self.tgt_shape[1]), anti_aliasing=True)
            image = cv2.resize(image, (self.tgt_shape[0], self.tgt_shape[1]))

            for gt_box in ground_truth_boxes:
                gt_box["bbox"].resize(kx, ky)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # reverse bgr to rgb
        image = img_yxc2cyx(image)  # replace channels position - YXC to CYX
        # src_shape = image.shape
        # image = np.reshape(image, (src_shape[2], src_shape[0], src_shape[1]))

        image = normalize_img_cyx(image)

        # image = image.astype(np.float32)
        # image = np.multiply(image, 1/255.)
        # image = np.subtract(image, 0.5)

        in_tensor = torch.from_numpy(image)
        # target_tensor = torch.zeros(size=(OUTPUT_TENSOR_C, OUTPUT_TENSOR_Y, OUTPUT_TENSOR_X), dtype=torch.float32)
        target = np.zeros(shape=(OUTPUT_TENSOR_Y, OUTPUT_TENSOR_X, OUTPUT_TENSOR_C), dtype=np.float32)

        for img_object in ground_truth_boxes:
            target_box = img_object["bbox"]
            target_class = img_object["class"]
            target_class_id = class2id[target_class]
            cell_x, cell_y, x_norm, y_norm, w_norm, h_norm = target_box.norm_coords
            if target[cell_y, cell_x, BBOX_C_POS] == 0:
                for i in range(BBOX_CNT):
                    target[cell_y, cell_x, BBOX_X_POS + i*BBOX_STRUCT_LEN] = x_norm
                    target[cell_y, cell_x, BBOX_Y_POS + i*BBOX_STRUCT_LEN] = y_norm
                    target[cell_y, cell_x, BBOX_W_POS + i*BBOX_STRUCT_LEN] = w_norm
                    target[cell_y, cell_x, BBOX_H_POS + i*BBOX_STRUCT_LEN] = h_norm
                    target[cell_y, cell_x, BBOX_C_POS + i*BBOX_STRUCT_LEN] = 1.
                target[cell_y, cell_x, CLASSES_DISTR_OFFSET + target_class_id] = 1
            # else:
            #     self.__msg__("Warning: __get_item__() -  skip objects, cell already contain target object")

        target = array_yxc2cyx(target)
        sample = {'input': in_tensor, 'target':
            torch.tensor(target)}

        return sample

    def get_labels(self):
        return self.index.get_labels()

    def __msg__(self, *txt):
        st = "<Dataset-VOC2007>: "
        for t in txt:
            st += t+" "
        print(st)


def decode_output_tensor(pred_tensor, conf_threshold=0.01):
    pred_tensor = pred_tensor.detach().numpy()
    pred_tensor = array_cyx2yxc(pred_tensor)
    # pred_tensor = np.reshape(pred_tensor.detach().numpy(), (OUTPUT_TENSOR_Y, OUTPUT_TENSOR_X, OUTPUT_TENSOR_C))
    objects_list = []

    for cell_y in range(OUTPUT_TENSOR_Y):
        for cell_x in range(OUTPUT_TENSOR_X):
            responses = []
            for bid in range(BBOX_CNT):
                conf = pred_tensor[cell_y, cell_x, BBOX_C_POS + BBOX_STRUCT_LEN * bid]
                if conf > conf_threshold:
                    responses.append((bid, conf))
            if len(responses):
                sorted_by_conf = sorted(responses, key=lambda tup: tup[1], reverse=True)
                box_id, conf = sorted_by_conf[0]
                bbox = BBox()
                norm_coords = (
                cell_x, cell_y, pred_tensor[cell_y, cell_x, BBOX_X_POS], pred_tensor[cell_y, cell_x, BBOX_Y_POS],
                pred_tensor[cell_y, cell_x, BBOX_W_POS], pred_tensor[cell_y, cell_x, BBOX_H_POS])
                bbox.set_norm(norm_coords)
                cell_probs = pred_tensor[cell_y, cell_x, CLASSES_DISTR_OFFSET:]
                max_class_id = cell_probs.argmax()
                class_name = "Unknown"
                if max_class_id in id2class.keys():
                    class_name = id2class[max_class_id]
                objects_list.append({"bbox": bbox.abs_coords, "conf": conf, "class": class_name,
                                     "class_id": max_class_id, "prob": cell_probs[max_class_id]})
    return objects_list


def decode_output_tensor_act(pred_tensor, conf_thresh=0.01):
    # act_pred = pred_tensor
    # act_pred[:CLASSES_DISTR_OFFSET, :, :] = torch.sigmoid(pred_tensor[:CLASSES_DISTR_OFFSET, :, :])
    # act_pred[CLASSES_DISTR_OFFSET:, :, :] = torch.nn.functional.softmax(pred_tensor[CLASSES_DISTR_OFFSET:,:,:], dim=0)
    act_pred = torch.sigmoid(pred_tensor)  # replace softmax at classes slice to sigmoid as well as for regressor
    return decode_output_tensor(act_pred, conf_thresh)


def show_sample(sample_dict):
    img = get_prediction_img(sample_dict)
    for obj in sample_dict["objects"]:
        _class = obj["class"]
        print("Object label: ", _class)
    show_image(img)
    pass


def show_image(im):
    # if im.shape[0] == 3:
    #     im = im.reshape(im.shape[1], im.shape[2], im.shape[0])
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)
    plt.show()
    pass


def get_prediction_img(sample_dict, convert_rgb2bgr=True):
    in_img = sample_dict["image"]
    if type(in_img) != np.ndarray:
        in_img = in_img.detach().numpy()
    # img = np.copy(in_img)
    img = copy.deepcopy(in_img)
    objs = sample_dict["objects"]

    img = denormalize_img_cyx(img)
    img = img_cyx2yxc(img)

    for obj in objs:
        _class = obj["class"]
        _bbox = obj["bbox"]
        _prob = obj["prob"]
        _conf = obj["conf"]
        _res_conf = _prob * _conf

        bx = [int(_bbox[0]), int(_bbox[1]), int(_bbox[0] + _bbox[2]), int(_bbox[1] + _bbox[3])]
        if bx[0] < 0:
            bx[0] = 0
        if bx[0] > img.shape[1] - 1:
            bx[0] = img.shape[1] - 1
        if bx[1] < 0:
            bx[1] = 0
        if bx[1] > img.shape[0] - 1:
            bx[1] = img.shape[0] - 1
        if bx[2] < 0:
            bx[2] = 0
        if bx[2] > img.shape[1] - 1:
            bx[2] = img.shape[1] - 1
        if bx[3] < 0:
            bx[3] = 0
        if bx[3] > img.shape[0] - 1:
            bx[3] = img.shape[0] - 1

        if _class in class2id.keys():
            clr = my_box_cv_colors[class2id[_class]]
        else:
            clr = [0, 0, 0]
        start = (bx[0], bx[1])
        stop = (bx[2], bx[3])
        box_c = clr  # rgb 2 cv-bgr

        thick = int(_res_conf * 6.0)
        if thick < 1:
            thick = 1
        # bbox
        img = cv2.rectangle(img=img, pt1=start, pt2=stop, color=box_c, thickness=thick)
        # label
        org_x, org_y = start[0] - 5, start[1] - 5
        if org_x < 0:
            org_x = 0
        if org_y < 0:
            org_y = 0
        img = cv2.putText(img, _class + ": " + "{0:.2f}".format(_res_conf), (org_x, org_y), FONT, FONT_SCALE, box_c, 1, cv2.LINE_AA)

    if convert_rgb2bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


############################################################################################


def visualize_dset_tensors(img, tgt):
    # print(decode_output_tensor(tgt))
    img = img.numpy()
    tgt = np.reshape(tgt.numpy(), (OUTPUT_TENSOR_Y, OUTPUT_TENSOR_X, OUTPUT_TENSOR_C))

    sample_dict = dict()
    sample_dict["image"] = img
    objects_list = []

    for cell_y in range(OUTPUT_TENSOR_Y):
        for cell_x in range(OUTPUT_TENSOR_X):
            if tgt[cell_y, cell_x, BBOX_C_POS]:
                cell_dict = dict()

                bbox = BBox()
                norm_coords = (cell_x, cell_y, tgt[cell_y, cell_x, BBOX_X_POS], tgt[cell_y, cell_x, BBOX_Y_POS],
                              tgt[cell_y, cell_x, BBOX_W_POS], tgt[cell_y, cell_x, BBOX_H_POS])
                bbox.set_norm(norm_coords)
                cell_probs = tgt[cell_y, cell_x, CLASSES_DISTR_OFFSET:]
                max_class_id = cell_probs.argmax()
                class_name = id2class[max_class_id]

                cell_dict["bbox"] = bbox.abs_coords
                cell_dict["class"] = class_name
                objects_list.append(cell_dict)

    sample_dict["objects"] = objects_list
    show_sample(sample_dict)
    pass
