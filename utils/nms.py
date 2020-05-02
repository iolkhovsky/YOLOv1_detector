from utils.bbox import BBox


def non_max_suppression(unfiltered_boxes, iou_threshold=0.7):
    """
    :param unfiltered_boxes: list of dicts - description of objects on image
    :param iou_threshold: threshold for iou
    ["bbox"] = BBox, ["conf"] = confidence, ["class"] = name, ["class_id"] = int_label, ["prob"] = prob
    :return: filtered_boxes
    """
    filtered_boxes = []
    for obj in unfiltered_boxes:
        obj_box, other_box = BBox(), BBox()
        obj_box.set_abs(obj["bbox"])
        discard = False
        for other in unfiltered_boxes:
            if obj["class"] == other["class"]:  # the same class
                other_box.set_abs(other["bbox"])
                if obj_box.iou(other_box) > iou_threshold:  # high IOU
                    if obj["conf"] * obj["prob"] < other["conf"] * obj["prob"]:  # low confidence
                        discard = True
            if not discard:
                filtered_boxes.append(obj)
    return filtered_boxes


# objects_list.append({"bbox": bbox.abs_coords, "conf": conf, "class": class_name,
#                      "class_id": max_class_id, "prob": cell_probs[max_class_id]})
#