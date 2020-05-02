import torch
import datetime
from os.path import isfile


repository_path = "/home/igor/models_checkpoints/"
cuda_available = torch.cuda.is_available()


def get_model_class_name(full_name):
    last_dot = full_name.rfind('.')
    last_kav = full_name.rfind("'")
    return full_name[last_dot + 1:last_kav]


def get_timestamp():
    stamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    stamp = stamp.replace(" ", "_")
    stamp = stamp.replace(":", "_")
    stamp = stamp.replace("-", "_")
    return stamp


def get_readable_timestamp():
    stamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return stamp


def compile_checkpoint_name(type_name, hint=None):
    file_name = repository_path
    if hint is not None:
        file_name += hint+"_"
    object_class_name = get_model_class_name(type_name)
    file_name += object_class_name + "_"
    file_name += get_timestamp()
    file_name += ".torchmodel"
    return file_name


def save_model(torch_model, hint=None):
    fname = compile_checkpoint_name(str(type(torch_model)), hint)
    # for the case when model stored on gpu device
    if cuda_available:
        torch_model.cpu()
    torch.save(torch_model, fname)
    if cuda_available:
        torch_model.cuda()
    print("Model '"+fname+"' successfully saved.")
    pass


def load_model(path):
    if repository_path not in path:
        print("Warning: Wrong path to models repository -> adding prefix automatically")
    path = repository_path + path
    torch_model = torch.load(path)
    return torch_model


def convert_model_to_cpu(path):
    model = load_model(path)
    model = model.cpu()
    save_model(model, "cpu")
    pass


def get_default_model_name():
    res = ""
    ini_path = "default_checkpoint_path.txt"
    if isfile(ini_path):
        with open(ini_path) as f:
            res = f.readline()
    return res.replace("\n", "")


