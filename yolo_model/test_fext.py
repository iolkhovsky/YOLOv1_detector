import torch
from torchvision.models import vgg11, mobilenet_v2
from yolo_model.yolo import YoloDetectorV1
from yolo_model.yolo import YoloDetectorV1MobNet


def test_models_weights(model, ref, start_par=0, cnt=None):
    ref_model = list(ref.parameters())
    test_model = list(model.parameters())
    if cnt is None:
        cnt = len(ref_model)
    for i in range(cnt):
        if not torch.equal(ref_model[i+start_par], test_model[i+start_par]):
            print("Error at parameter ", i, ref_model[i+start_par].shape, test_model[i+start_par].shape)
            return False
    return True


def test_vgg11_fext():
    print("YoloDetectorV1 feature extractor test (VGG11)")
    model = YoloDetectorV1()
    probe = vgg11(pretrained=True)
    res = test_models_weights(model=probe, ref=model, start_par=0, cnt=16)
    print("Comparsion of weights in models: ")
    if res:
        print("OK")
        return True
    else:
        print("Not equal")
        return False


def test_mobilenetv2_fext():
    print("YoloDetectorV1MobNet feature extractor test (MobilenetV2)")
    model = YoloDetectorV1MobNet()
    probe = mobilenet_v2(pretrained=True)
    res = test_models_weights(model=probe, ref=model, start_par=0, cnt=16)
    print("Comparsion of weights in models: ")
    if res:
        print("OK")
        return True
    else:
        print("Not equal")
        return False
