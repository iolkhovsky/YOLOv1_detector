import torch
from yolo_model.feature_ext import *


class YoloClfRegrHeadVgg11(nn.Module):

    def __init__(self):
        super(YoloClfRegrHeadVgg11, self).__init__()
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=7*7*512, out_features=4096)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=7*7*30)
        pass

    def forward(self, x):
        x = x.view(-1, 7*7*512)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = x.view(-1, 30, 7, 7)
        return x


class YoloClfRegrHeadMobnet2(nn.Module):

    def __init__(self):
        super(YoloClfRegrHeadMobnet2, self).__init__()
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=7*7*1280, out_features=4096)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=7*7*30)
        pass

    def forward(self, x):
        x = x.view(-1, 7*7*1280)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = x.view(-1, 30, 7, 7)
        return x


class YoloDetectorV1(nn.Module):

    def __init__(self, pretrained=True, fext_requires_grad=False):
        super(YoloDetectorV1, self).__init__()
        self.feat_ext = VGG11FeatureExtractor(pretrained=pretrained, requires_grad=fext_requires_grad)
        self.head = YoloClfRegrHeadVgg11()
        pass

    def forward(self, x):
        x = self.feat_ext(x)
        x = self.head(x)
        return x


class YoloDetectorV1MobNet(nn.Module):

    def __init__(self, pretrained=True, fext_requires_grad=False):
        super(YoloDetectorV1MobNet, self).__init__()
        self.feat_ext = MobilenetV2FeatureExtractor(pretrained=pretrained, requires_grad=fext_requires_grad)
        self.head = YoloClfRegrHeadMobnet2()
        pass

    def forward(self, x):
        x = self.feat_ext(x)
        x = self.head(x)
        return x
