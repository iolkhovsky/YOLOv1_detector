#!/usr/bin/python3

from pascal_voc.voc_detection import *
from yolo_model.yolo import *
from yolo_model.models_storage import *
from utils.img_proc import *
from utils.nms import *
import cv2
import sys


model = None
def_model_path = get_default_model_name()
if def_model_path == "":
    model = YoloDetectorV1(pretrained=True)
else:
    model = load_model(def_model_path)
use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()
model.eval()


src = 0
if len(sys.argv) > 1:
    src_str = sys.argv[1]
    try:
        src = int(src_str)
    except ValueError:
        src = src_str
    if isinstance(src, str):
        print("Selected video file: ", src)
    else:
        print("Selected web-camera id: ", src)

cap = cv2.VideoCapture(src)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(224, 224))
    # image = image.astype(np.float32)
    # image = np.multiply(image, 1 / 255.)
    # image = np.subtract(image, 0.5)
    image = array_yxc2cyx(image)
    image = normalize_img_cyx(image)

    in_tensor = torch.tensor(image).reshape(shape=(-1, 3, 224, 224))
    if use_gpu:
        in_tensor = in_tensor.cuda()
    prediction_batch = model.forward(in_tensor)
    if use_gpu:
        prediction_batch = prediction_batch.cpu()
    detections = decode_output_tensor_act(prediction_batch[0], 0.5)
    detections_after_nms = non_max_suppression(detections, iou_threshold=0.2)

    img_with_boxes = get_prediction_img({"image": image, "objects": detections}, convert_rgb2bgr=True)
    # img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
    img_with_boxes = cv2.resize(img_with_boxes, dsize=(224*4, 224*4))
    print(detections)

    img_with_boxes_nms = get_prediction_img({"image": image, "objects": detections_after_nms})
    # img_with_boxes_nms = cv2.cvtColor(img_with_boxes_nms, cv2.COLOR_RGB2BGR)
    img_with_boxes_nms = cv2.resize(img_with_boxes_nms, dsize=(4*224, 4*224))
    # print(detections_after_nms)

    # Display the resulting frame
    cv2.imshow('stream', img_with_boxes)
    cv2.imshow('frame nms', img_with_boxes_nms)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
