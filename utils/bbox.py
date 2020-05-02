import numpy as np
from yolo_model.yolo_constraints import *
import numbers
import decimal

DEF_IMG_SIZE = (INPUT_TENSOR_X, INPUT_TENSOR_Y)
DEF_CELLS_CNT = (OUTPUT_TENSOR_X, OUTPUT_TENSOR_Y)


def is_number(probe):
    return [isinstance(probe, numbers.Number) for x in (0, 0.0, 0j, decimal.Decimal(0))]


def get_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersection_x0 = max(x1, x2)
    intersection_x1 = min(x1+w1, x2+w2)
    if intersection_x1 <= intersection_x0:
        return 0.
    intersection_y0 = max(y1, y2)
    intersection_y1 = min(y1+h1, y2+h2)
    if intersection_y1 <= intersection_y0:
        return 0.
    intersection_area = (intersection_x1 - intersection_x0) * (intersection_y1 - intersection_y0)
    box1_area = w1 * h1
    box2_area = w2 * h2
    return intersection_area / (box1_area + box2_area - intersection_area)


class BBox:

    def __init__(self, img_size=DEF_IMG_SIZE, cells=DEF_CELLS_CNT):
        '''
        :param img_size: (x_sz, y_sz) - reference size of image (integers)
        :param cells: (x_cnt, y_cnt) - reference
        '''
        self.img_sz = img_size
        self.cells = cells
        # default
        self.abs_coords = None
        self.norm_coords = None
        self.valid = False
        self.valid_for_img = False
        pass

    def set_abs(self, coords):
        '''
        :param coords: (x0, y0, w, h) - reference strobe in abs coord system (integers)
        :return:
        '''
        self.abs_coords = coords
        self.__update_norm__()
        pass

    def set_norm(self, coords):
        '''
        :param coords: (cx, cy, x, y, w, h) - reference norm strobe in rel system(cx,cy - int, other - float)
        :return:
        '''
        self.norm_coords = coords
        self.__update_abs__()
        pass

    def resize(self, kx, ky):
        self.valid = self.__abs_coord_valid()
        if self.valid:
            x, y, w, h = self.abs_coords
            self.abs_coords = (x * kx, y * ky, w * kx, h * ky)
        pass

    def is_valid(self):
        self.valid = self.__abs_coord_valid()
        return self.valid

    def iou(self, other):
        if self.valid and other.is_valid():
            return get_iou(self.abs_coords, other.abs_coords)
        else:
            return 0.

    def __update_norm__(self):
        self.valid = self.__abs_coord_valid()
        if self.valid:
            x0, y0, w, h = self.abs_coords
            w_norm = w * 1.0 / self.img_sz[0]
            h_norm = h * 1.0 / self.img_sz[1]

            cell_size_x = self.img_sz[0] * 1.0 / self.cells[0]
            cell_size_y = self.img_sz[1] * 1.0 / self.cells[1]
            center_x = x0 + w * 0.5
            center_y = y0 + h * 0.5
            cell_x = int(center_x / cell_size_x)
            cell_y = int(center_y / cell_size_y)

            x_norm = (center_x - cell_x * cell_size_x) / cell_size_x
            y_norm = (center_y - cell_y * cell_size_y) / cell_size_y

            w_norm = np.sqrt(w_norm)
            h_norm = np.sqrt(h_norm)

            self.norm_coords = (cell_x, cell_y, x_norm, y_norm, w_norm, h_norm)
        else:
            print("Error!: Can update norm coord - abs coord invalid!\n")
        pass

    def __update_abs__(self):
        self.valid = self.__norm_coord_valid()
        if self.valid:
            cx, cy, x, y, w, h = self.norm_coords
            abs_w = np.power(w, 2) * self.img_sz[0]
            abs_h = np.power(h, 2) * self.img_sz[1]

            cell_size_x = self.img_sz[0] * 1.0 / self.cells[0]
            cell_size_y = self.img_sz[1] * 1.0 / self.cells[1]
            abs_center_x = (cx + x) * cell_size_x
            abs_center_y = (cy + y) * cell_size_y
            abs_x = abs_center_x - 0.5 * abs_w
            abs_y = abs_center_y - 0.5 * abs_h

            self.abs_coords = (abs_x, abs_y, abs_w, abs_h)
        pass

    def __abs_coord_valid(self):
        if self.abs_coords is None:
            return False
        elif len(self.abs_coords) != 4:
            return False
        else:
            for i in range(4):
                if not is_number(self.abs_coords[i]):
                    return False
        return True

    def __norm_coord_valid(self):
        if self.norm_coords is None:
            return False
        elif len(self.norm_coords) != 6:
            return False
        else:
            for i in range(6):
                if not is_number(self.norm_coords[i]):
                    return False
        return True


# IOU testing
#
# a = BBox(2, 3, 2, 2)
# b = BBox(3, 4, 2, 2)
# c = BBox(2, 3, 2, 2)
# d = BBox(10, -9, 2, 2)
#
# probe = [b, c, d]
# for test in probe:
#     print("IOU: ", a.iou(test))
