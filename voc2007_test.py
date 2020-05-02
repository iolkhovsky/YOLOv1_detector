from pascal_voc.voc_detection import VocDetection, show_sample
from torch.utils.data import DataLoader

dataset_root = "/home/igor/scripts/yolo_torch/VOC_2007"
subset = "trainval"
out_tensor_shape = (300, 300)

train_dataset = VocDetection(dataset_root, subset, out_tensor_shape)
train_dataloader = DataLoader(train_dataset, shuffle=True)

samples_limit = 10

cnt = 0
for sample_id, sample in enumerate(train_dataloader):
    print(sample_id, sample["objects"])
    if cnt == samples_limit:
        break
    cnt += 1

sample = train_dataset[cnt]
show_sample(sample)
