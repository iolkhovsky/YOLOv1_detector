import numpy as np
import torch


torchvision_norm_mean = [0.485, 0.456, 0.406]
torchvision_norm_std = [0.229, 0.224, 0.225]


def img_cyx2yxc(img):
    if type(img) != np.ndarray:
        raise ValueError("Not a np.ndarray as input frame!")
    csz, ysz, xsz = img.shape
    out = np.zeros(shape=(ysz, xsz, csz), dtype=np.uint8)
    for ch in range(csz):
        out[:, :, ch] = img[ch, :, :]
    return out


def img_yxc2cyx(img):
    if type(img) != np.ndarray:
        raise ValueError("Not a np.ndarray as input frame!")
    ysz, xsz, csz = img.shape
    out = np.zeros(shape=(csz, ysz, xsz), dtype=np.uint8)
    for ch in range(csz):
        out[ch, :, :] = img[:, :, ch]
    return out


def array_cyx2yxc(tns):
    if type(tns) != np.ndarray:
        raise ValueError("Not a np.ndarray as input tensor!")
    csz, ysz, xsz = tns.shape
    out = np.zeros(shape=(ysz, xsz, csz), dtype=np.float32)
    for ch in range(csz):
        out[:, :, ch] = tns[ch, :, :]
    return out


def array_yxc2cyx(tns):
    if type(tns) != np.ndarray:
        raise ValueError("Not a np.ndarray as input tensor!")
    ysz, xsz, csz = tns.shape
    out = np.zeros(shape=(csz, ysz, xsz), dtype=np.float32)
    for ch in range(csz):
        out[ch, :, :] = tns[:, :, ch]
    return out


def tensor_cyx2yxc(tns):
    # if type(tns) != torch.tensor:
    #     raise ValueError("Not a torch.tensor as input tensor!")
    if len(tns.shape) == 4:  # batch
        b, c, y, x = tns.shape
        out = torch.zeros(size=(b, y, x, c), dtype=torch.float32, device=tns.device)
        for ch in range(c):
            out[:, :, :, ch] = tns[:, ch, :, :]
        return out
    elif len(tns.shape) == 3:  # one tensor
        c, y, x = tns.shape
        out = torch.zeros(size=(y, x, c), dtype=torch.float32, device=tns.device)
        for ch in range(c):
            out[:, :, ch] = tns[ch, :, :]
        return out
    else:
        raise ValueError("Wrong input")


def tensor_yxc2cyx(tns):
    if type(tns) != torch.tensor:
        raise ValueError("Not a torch.tensor as input tensor!")
    y, x, c = tns.shape[-3:]
    out = torch.zeros(size=(c, y, x), device=tns.get_device())
    for ch in range(c):
        out[ch, :, :] = tns[:, :, ch]
    return out


# all pretrained models in torchvision model zoo have the same normalization procedure:
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def normalize_img_cyx(img):
    img = img.astype(dtype=np.float32)
    img = np.divide(img, 255.)  # 0...255 -> 0...1.
    r, g, b = img[0], img[1], img[2]
    r = np.divide(r - torchvision_norm_mean[0], torchvision_norm_std[0])
    g = np.divide(g - torchvision_norm_mean[1], torchvision_norm_std[1])
    b = np.divide(b - torchvision_norm_mean[2], torchvision_norm_std[2])
    img[0] = r
    img[1] = g
    img[2] = b
    return img


def denormalize_img_cyx(img):
    r, g, b = img[0], img[1], img[2]
    r = np.multiply(r, torchvision_norm_std[0]) + torchvision_norm_mean[0]
    g = np.multiply(g, torchvision_norm_std[1]) + torchvision_norm_mean[1]
    b = np.multiply(b, torchvision_norm_std[2]) + torchvision_norm_mean[2]
    img[0] = r
    img[1] = g
    img[2] = b
    img = np.multiply(img, 255.)  # 0...1. -> 0...255.
    img = img.astype(np.uint8)
    return img


