# Core
from os import listdir
from os.path import isfile, isdir, join
import xml.etree.ElementTree as ET
# Interface
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.bbox import BBox
import random


class ImageDescription:
    
    def __init__(self, path=None, size=None, objs=[]):
        self.img_path = path
        self.img_size = size
        self.objects_list = objs
        pass
    
    def set(self, path, size, objs):
        self.img_path = path
        self.img_size = size
        self.objects_list = objs
        pass
    
    def get_path(self):
        return self.img_path
    
    def get_img_size(self):
        return self.img_size
    
    def get_objects(self):
        return self.objects_list
    
    def get_objects_cnt(self):
        return len(self.objects_list)


class DatasetIndex:
    
    def __init__(self, root_path=None, index_name=None, size_limit=None):
        self.root = root_path
        self.ok = False
        self.sz_limit = size_limit
        if size_limit is not None:
            self.sz_limit = int(size_limit)
        self.img_descriptions = None
        self.annotations = None
        self.index_name = index_name
        self.__dset_classes = set()
        self.load()
        pass
    
    def load(self, root_path=None):
        self.ok = True
        if root_path is not None:
            self.root = root_path
        if self.root is not None and isdir(self.root):
            self.annotations = self.__get_files_list(path=self.root+"/Annotations", filter_key=".xml")
            if self.sz_limit is not None:
                # shuffle annotations list
                random.shuffle(self.annotations)
                # get clipped list
                self.annotations = self.annotations[:self.sz_limit]
            self.__compile_descriptions()
        else:
            self.ok = False
            
        if self.ok:
            self.__msg("Index successfully updated")
        else:
            self.__msg("Error: invalid root path ot internal structure")
        pass
    
    def __compile_descriptions(self):
        self.img_descriptions = []
        cnt = 0
        for ann in self.annotations:
            xml_data = self.__get_img_dict(ann)
            new_labels = set()
            for obj in xml_data["objects"]:
                new_labels.add(obj["class"])
            self.img_descriptions.append(ImageDescription(xml_data["abs_path"],
                                                         xml_data["size"],
                                                         xml_data["objects"]))
            self.__dset_classes = self.__dset_classes.union(new_labels)
            cnt += 1
        self.__msg("Found "+str(cnt)+" annotations")
        pass
    
    def __get_img_dict(self, annotation):
        out = {}
        tree = ET.parse(annotation)
        root = tree.getroot()
        out["objects"] = []

        # first find filename and image size
        for child in root:
            if child.tag == "filename":
                out["abs_path"] = self.root + "JPEGImages/" + child.text
            if child.tag == "size":
                w = None
                h = None
                c = None
                for _child in child:
                    if _child.tag == "width":
                        w = int(_child.text)
                    elif _child.tag == "height":
                        h = int(_child.text)
                    elif _child.tag == "depth":
                        c = int(_child.text)
                out["size"] = (c, h, w)

        # then find bboxes
        for child in root:
            if child.tag == "object":
                objdesc = {}
                for _child in child:
                    if _child.tag == "name":
                        objdesc["class"] = _child.text
                    if _child.tag == "bndbox":
                        x0, x1, y0, y1 = None, None, None, None
                        for _par in _child:
                            if _par.tag == "xmin":
                                x0 = int(float(_par.text))
                            elif _par.tag == "ymin":
                                y0 = int(float(_par.text))
                            elif _par.tag == "xmax":
                                x1 = int(float(_par.text))
                            elif _par.tag == "ymax":
                                y1 = int(float(_par.text))
                        buf = BBox(img_size=(out["size"][2], out["size"][1]))
                        buf.set_abs((x0, y0, x1 - x0, y1 - y0))
                        objdesc["bbox"] = buf  # x, y, w, h
                out["objects"].append(objdesc)
        return out

    @staticmethod
    def __get_files_list(path, filter_key=None):
        onlyfiles = None
        if filter_key is None:
            onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        else:
            onlyfiles = [join(path, f) for f in listdir(path) if (isfile(join(path, f)) and filter_key in f)]
        return onlyfiles
    
    def __msg(self, text):
        print("<DatasetIndex>: "+self.index_name+" "+text)
        
    def get_data_images(self):
        return self.img_descriptions
    
    def get_img_description(self, idx):
        return self.img_descriptions[idx]

    def get_labels(self):
        return self.__dset_classes

    def get_size(self):
        return len(self.annotations)


class VocIndex:
    
    def __init__(self, root_path, size_limit=None):
        self.root = root_path
        self.trainval_path = root_path + "/trainval/"
        self.test_path = root_path + "/test/"
        self.trainval = DatasetIndex(self.trainval_path, "train_val", size_limit=size_limit)
        self.test = DatasetIndex(self.test_path, "test", size_limit=size_limit)
        pass
    
    def show_summary(self):
        self.__msg("Trainval index consist of " + str(len(self.trainval.GetDataImages())) + " objects")
        self.__msg("Test index consist of " + str(len(self.test.GetDataImages())) + " objects")
        pass

    @staticmethod
    def __msg(self, text):
        print("<VocDataset>: "+ text)
        pass
    
    def get_trainval(self):
        return self.trainval
    
    def get_test(self):
        return self.test
    

def show_dataset_image(image_descriptor):
    path = image_descriptor.GetPath()
    size = image_descriptor.GetImgSize()
    objs = image_descriptor.GetObjects()
    print("Image file abs path ", path)
    print("Size of image: ", size)
    print("Objects ",len(objs))
    img = pyplot.imread(path)
    
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(img)
    # Create a Rectangle patch
    for obj in objs:
        _class = obj["class"]
        _bbox = obj["bbox"]
        rect = patches.Rectangle((_bbox[0], _bbox[1]), _bbox[2], _bbox[3],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        print("Object label: ", _class)
    plt.show()
    pass

