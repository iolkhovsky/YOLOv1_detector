from pascal_voc.voc_detection import VocDetection, decode_output_tensor, decode_output_tensor_act
from torch.utils.data import DataLoader
from yolo_model.models_storage import *
from yolo_model.yolo import *
from metrics.detection_metrics import *


# pascal_voc_root = "/home/igor/datasets/VOC_2007/"
pascal_voc_root = "/home/igor/datasets/VOC_2012/"

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

test_dataset = VocDetection(pascal_voc_root, subset="trainval", target_shape=(INPUT_TENSOR_X, INPUT_TENSOR_Y))
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

det_objs_list = []
truth_objs_list = []

for sample in test_dataloader:
    image_batch = sample["input"]
    target_batch = sample["target"]

    if use_gpu:
        image_batch = image_batch.cuda()
        target_batch = target_batch.cuda()

    prediction_batch = model.forward(image_batch)

    if use_gpu:
        prediction_batch = prediction_batch.cpu()
        target_batch = target_batch.cpu()

    for idx in range(prediction_batch.shape[0]):
        print(idx, " of ", prediction_batch.shape[0])
        det_objs = decode_output_tensor_act(prediction_batch[idx], conf_thresh=0.1)
        det_objs = non_max_suppression(det_objs, 0.5)
        truth_objs = decode_output_tensor(target_batch[idx])
        det_objs_list.append(det_objs)
        truth_objs_list.append(truth_objs)
    break

map_value = mean_average_precision(det_objs_list, truth_objs_list)

for k in map_value.keys():
    print(k, ": ", map_value[k])

print("================================================")
print("mAP: ", map_value["map"], " on subset of size: ", len(det_objs_list))
print("Completed.")
