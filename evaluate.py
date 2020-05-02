from pascal_voc.voc_detection import VocDetection, decode_output_tensor, decode_output_tensor_act, get_prediction_img
from torch.utils.data import DataLoader
from yolo_model.yolo import *
from yolo_model.models_storage import *
from metrics.detection_metrics import *
from os.path import isdir
from os import mkdir
import shutil
import cv2

cuda_available = torch.cuda.is_available()

# pascal_voc_root = "/home/igor/datasets/VOC_2007/"
pascal_voc_root = "/home/igor/datasets/VOC_2012/"
output_imgs = "eval_output"

if isdir(output_imgs):
    shutil.rmtree(output_imgs)
mkdir(output_imgs)

model = None
def_model_path = get_default_model_name()
if def_model_path == "":
    model = YoloDetectorV1(pretrained=True)
else:
    model = load_model(def_model_path)

if cuda_available:
    model.cuda()
else:
    model.cpu()
model.eval()

test_dataset = VocDetection(pascal_voc_root, subset="trainval", target_shape=(INPUT_TENSOR_X, INPUT_TENSOR_Y))
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

det_objs_list = []
truth_objs_list = []

for sample in test_dataloader:
    image_batch = sample["input"]
    target_batch = sample["target"]
    if cuda_available:
        image_batch = image_batch.cuda()
    prediction_batch = model.forward(image_batch)
    if cuda_available:
        image_batch = image_batch.cpu()
        prediction_batch = prediction_batch.cpu()

    for idx in range(prediction_batch.shape[0]):
        print(idx, " of ", prediction_batch.shape[0])
        det_objs = decode_output_tensor_act(prediction_batch[idx], conf_thresh=0.1)
        det_objs = non_max_suppression(det_objs, 0.5)
        truth_objs = decode_output_tensor(target_batch[idx])
        det_objs_list.append(det_objs)
        truth_objs_list.append(truth_objs)
        # show_sample({"image": image_batch[idx], "objects": det_objs})
        # show_sample({"image": image_batch[idx], "objects": truth_objs})
        pr_img = get_prediction_img({"image": image_batch[idx], "objects": det_objs})
        tgt_img = get_prediction_img({"image": image_batch[idx], "objects": truth_objs})
        # pr_img = cv2.cvtColor(pr_img, cv2.COLOR_RGB2BGR)
        # tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_imgs + "/" + str(idx) + "_predicted.jpg", pr_img)
        cv2.imwrite(output_imgs + "/" + str(idx) + "_target.jpg", tgt_img)
    opt_conf = get_optimal_threshold(det_objs_list, truth_objs_list)
    break

map_value = mean_average_precision(det_objs_list, truth_objs_list)

outbuf = ""
for k in map_value.keys():
    outbuf += k+": "+str(map_value[k]) + "\n"

with open("AP_summary.txt", "w") as f:
    f.write(outbuf)

print("mAP: ", map_value["map"], " on subset of size: ", len(det_objs_list))
print("Completed.")
