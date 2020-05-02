import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch


vgg11_par_par_index = {"conv1.weight": 0, "conv1.bias": 1, "conv2.weight": 2, "conv2.bias": 3,
                       "conv3.weight": 4, "conv3.bias": 5, "conv4.weight": 6, "conv4.bias": 7,
                       "conv5.weight": 8, "conv5.bias": 9, "conv6.weight": 10, "conv6.bias": 11,
                       "conv7.weight": 12, "conv7.bias": 13, "conv8.weight": 14, "conv8.bias": 15}


class VGG11FeatureExtractor(nn.Module):

    def __init__(self, pretrained=True, requires_grad=False):
        super(VGG11FeatureExtractor, self).__init__()

        self.pretrained = pretrained
        self.requires_grad = requires_grad

        # architecture

        # input tensor - 3x224x224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)  # 64x224x224
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                                          # 64x112x112
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # 128x112x112
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                                              # 128x56x56
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # 256x56x56
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)  # 256x56x56
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)                                             # 256x28x28
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)  # 512x28x28
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)  # 512x28x28
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)                                             # 512x14x14
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)  # 512x14x14
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)  # 512x14x14
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2)                                             # 512x7x7

        # initialization

        if self.pretrained is not None:
            if self.pretrained:
                vgg11 = models.vgg11(pretrained=True)
                w_pretrained = list(vgg11.parameters())
                self.conv1.weight = w_pretrained[vgg11_par_par_index["conv1.weight"]]
                self.conv1.bias = w_pretrained[vgg11_par_par_index["conv1.bias"]]
                for param in self.conv1.parameters():
                    param.requires_grad = requires_grad
                self.conv2.weight = w_pretrained[vgg11_par_par_index["conv2.weight"]]
                self.conv2.bias = w_pretrained[vgg11_par_par_index["conv2.bias"]]
                for param in self.conv2.parameters():
                    param.requires_grad = requires_grad
                self.conv3.weight = w_pretrained[vgg11_par_par_index["conv3.weight"]]
                self.conv3.bias = w_pretrained[vgg11_par_par_index["conv3.bias"]]
                for param in self.conv3.parameters():
                    param.requires_grad = requires_grad
                self.conv4.weight = w_pretrained[vgg11_par_par_index["conv4.weight"]]
                self.conv4.bias = w_pretrained[vgg11_par_par_index["conv4.bias"]]
                for param in self.conv4.parameters():
                    param.requires_grad = requires_grad
                self.conv5.weight = w_pretrained[vgg11_par_par_index["conv5.weight"]]
                self.conv5.bias = w_pretrained[vgg11_par_par_index["conv5.bias"]]
                for param in self.conv5.parameters():
                    param.requires_grad = requires_grad
                self.conv6.weight = w_pretrained[vgg11_par_par_index["conv6.weight"]]
                self.conv6.bias = w_pretrained[vgg11_par_par_index["conv6.bias"]]
                for param in self.conv6.parameters():
                    param.requires_grad = requires_grad
                self.conv7.weight = w_pretrained[vgg11_par_par_index["conv7.weight"]]
                self.conv7.bias = w_pretrained[vgg11_par_par_index["conv7.bias"]]
                for param in self.conv7.parameters():
                    param.requires_grad = requires_grad
                self.conv8.weight = w_pretrained[vgg11_par_par_index["conv8.weight"]]
                self.conv8.bias = w_pretrained[vgg11_par_par_index["conv8.bias"]]
                for param in self.conv8.parameters():
                    param.requires_grad = requires_grad
        pass

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool4(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool6(F.relu(self.conv6(x)))
        x = F.relu(self.conv7(x))
        x = self.pool8(F.relu(self.conv8(x)))
        return x


class ConvBNRelu(nn.Module):

    def __init__(self, in_chan, out_chan, kernel=3, stride=2, pad=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=(kernel, kernel), stride=(stride, stride),
                              padding=(pad, pad), bias=False)
        self.bn = nn.BatchNorm2d(out_chan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.ReLU6(inplace=True)
        pass

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def init_from_list(self, pars):
        self.conv.weight = pars[0]
        self.bn.weight = pars[1]
        self.bn.bias = pars[2]
        pass

    def enable_grad(self, en):
        self.conv.weight.requires_grad = en
        self.bn.weight.requires_grad = en
        self.bn.bias.requires_grad = en
        pass


class Bottleneck(nn.Module):

    def __init__(self, in_chan, t_factor, out_chan, stride):
        super(Bottleneck, self).__init__()
        self.t = t_factor
        self.cin = in_chan
        self.cout = out_chan
        self.s = stride
        # input pointwise
        self.conv_pw = None
        self.bn_pw = None
        self.act_pw = None
        if t_factor != 1:
            self.conv_pw = nn.Conv2d(in_chan, in_chan * t_factor, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.bn_pw = nn.BatchNorm2d(in_chan * t_factor, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.act_pw = nn.ReLU6(inplace=True)
        # depthwise
        self.conv_dw = nn.Conv2d(in_chan * t_factor, in_chan * t_factor, kernel_size=(3, 3), stride=(stride, stride),
                                 groups=in_chan * t_factor, bias=False, padding=(1, 1))
        self.bn_dw = nn.BatchNorm2d(in_chan * t_factor, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act_dw = nn.ReLU6(inplace=True)
        # output pointwise
        self.conv_pw_out = nn.Conv2d(in_chan * t_factor, out_chan, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn_pw_out = nn.BatchNorm2d(out_chan, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.act_pw_out = nn.ReLU6(inplace=True)
        pass

    def forward(self, x):
        in_act = x
        # first pointwise convolution
        x_pw = self.act_pw(self.bn_pw(self.conv_pw(x))) if self.t != 1 else x
        x_dw = self.act_dw(self.bn_dw(self.conv_dw(x_pw)))
        # x_pw_o = self.act_pw_out(self.bn_pw_out(self.conv_pw_out(x_dw)))
        x_pw_o = self.bn_pw_out(self.conv_pw_out(x_dw))
        # residual connection
        out_act = torch.add(x_pw_o, in_act) if (self.cin == self.cout) and (self.s == 1) else x_pw_o
        return out_act

    def init_from_list(self, pars):
        if self.t != 1:
            self.conv_pw.weight = pars[0]
            self.bn_pw.weight = pars[1]
            self.bn_pw.bias = pars[2]
            self.conv_dw.weight = pars[3]
            self.bn_dw.weight = pars[4]
            self.bn_dw.bias = pars[5]
            self.conv_pw_out.weight = pars[6]
            self.bn_pw_out.weight = pars[7]
            self.bn_pw_out.bias = pars[8]
        else:
            self.conv_dw.weight = pars[0]
            self.bn_dw.weight = pars[1]
            self.bn_dw.bias = pars[2]
            self.conv_pw_out.weight = pars[3]
            self.bn_pw_out.weight = pars[4]
            self.bn_pw_out.bias = pars[5]
        pass

    def enable_grad(self, en):
        if self.t != 1:
            self.conv_pw.weight.requires_grad = en
            self.bn_pw.weight.requires_grad = en
            self.bn_pw.bias.requires_grad = en
        self.conv_dw.weight.requires_grad = en
        self.bn_dw.weight.requires_grad = en
        self.bn_dw.bias.requires_grad = en
        self.conv_pw_out.weight.requires_grad = en
        self.bn_pw_out.weight.requires_grad = en
        self.bn_pw_out.bias.requires_grad = en
        pass


class MobilenetV2FeatureExtractor(nn.Module):

    def __init__(self, pretrained=True, requires_grad=False):
        super(MobilenetV2FeatureExtractor, self).__init__()

        self.pretrained = pretrained
        self.requires_grad = requires_grad

        # architecture

        # input tensor - 3x224x224
        self.conv0 = ConvBNRelu(in_chan=3, out_chan=32, kernel=3, stride=2, pad=1)          # 32x112x112
        self.bottleneck1 = Bottleneck(in_chan=32, t_factor=1, out_chan=16, stride=1)        # 16x112x112
        self.bottleneck2 = Bottleneck(in_chan=16, t_factor=6, out_chan=24, stride=2)        # 24x56x56
        self.bottleneck3 = Bottleneck(in_chan=24, t_factor=6, out_chan=24, stride=1)        # 24x56x56
        self.bottleneck4 = Bottleneck(in_chan=24, t_factor=6, out_chan=32, stride=2)        # 32x28x28
        self.bottleneck5 = Bottleneck(in_chan=32, t_factor=6, out_chan=32, stride=1)        # 32x28x28
        self.bottleneck6 = Bottleneck(in_chan=32, t_factor=6, out_chan=32, stride=1)        # 32x28x28
        self.bottleneck7 = Bottleneck(in_chan=32, t_factor=6, out_chan=64, stride=2)        # 64x14x14
        self.bottleneck8 = Bottleneck(in_chan=64, t_factor=6, out_chan=64, stride=1)        # 64x14x14
        self.bottleneck9 = Bottleneck(in_chan=64, t_factor=6, out_chan=64, stride=1)        # 64x14x14
        self.bottleneck10 = Bottleneck(in_chan=64, t_factor=6, out_chan=64, stride=1)       # 64x14x14
        self.bottleneck11 = Bottleneck(in_chan=64, t_factor=6, out_chan=96, stride=1)       # 96x14x14
        self.bottleneck12 = Bottleneck(in_chan=96, t_factor=6, out_chan=96, stride=1)       # 96x14x14
        self.bottleneck13 = Bottleneck(in_chan=96, t_factor=6, out_chan=96, stride=1)       # 96x14x14
        self.bottleneck14 = Bottleneck(in_chan=96, t_factor=6, out_chan=160, stride=2)      # 160x7x7
        self.bottleneck15 = Bottleneck(in_chan=160, t_factor=6, out_chan=160, stride=1)     # 160x7x7
        self.bottleneck16 = Bottleneck(in_chan=160, t_factor=6, out_chan=160, stride=1)     # 160x7x7
        self.bottleneck17 = Bottleneck(in_chan=160, t_factor=6, out_chan=320, stride=1)     # 320x7x7
        self.conv18 = ConvBNRelu(in_chan=320, out_chan=1280, kernel=1, stride=1, pad=0)            # 1280x7x7

        self.pars_count_per_layer = [3, 6]
        self.pars_count_per_layer.extend([9] * 16)
        self.pars_count_per_layer.extend([3])

        # initialization

        if self.pretrained is not None:
            if self.pretrained:
                mobilenetv2 = models.mobilenet_v2(pretrained=True)
                w_pretrained = list(mobilenetv2.parameters())

                start = 0
                stop = start + self.pars_count_per_layer[0]
                sublist = w_pretrained[start: stop]
                self.conv0.init_from_list(sublist)
                self.conv0.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[1]
                sublist = w_pretrained[start: stop]
                self.bottleneck1.init_from_list(sublist)
                self.bottleneck1.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[2]
                sublist = w_pretrained[start: stop]
                self.bottleneck2.init_from_list(sublist)
                self.bottleneck2.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[3]
                sublist = w_pretrained[start: stop]
                self.bottleneck3.init_from_list(sublist)
                self.bottleneck3.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[4]
                sublist = w_pretrained[start: stop]
                self.bottleneck4.init_from_list(sublist)
                self.bottleneck4.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[5]
                sublist = w_pretrained[start: stop]
                self.bottleneck5.init_from_list(sublist)
                self.bottleneck5.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[6]
                sublist = w_pretrained[start: stop]
                self.bottleneck6.init_from_list(sublist)
                self.bottleneck6.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[7]
                sublist = w_pretrained[start: stop]
                self.bottleneck7.init_from_list(sublist)
                self.bottleneck7.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[8]
                sublist = w_pretrained[start: stop]
                self.bottleneck8.init_from_list(sublist)
                self.bottleneck8.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[9]
                sublist = w_pretrained[start: stop]
                self.bottleneck9.init_from_list(sublist)
                self.bottleneck9.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[10]
                sublist = w_pretrained[start: stop]
                self.bottleneck10.init_from_list(sublist)
                self.bottleneck10.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[11]
                sublist = w_pretrained[start: stop]
                self.bottleneck11.init_from_list(sublist)
                self.bottleneck11.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[12]
                sublist = w_pretrained[start: stop]
                self.bottleneck12.init_from_list(sublist)
                self.bottleneck12.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[13]
                sublist = w_pretrained[start: stop]
                self.bottleneck13.init_from_list(sublist)
                self.bottleneck13.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[14]
                sublist = w_pretrained[start: stop]
                self.bottleneck14.init_from_list(sublist)
                self.bottleneck14.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[15]
                sublist = w_pretrained[start: stop]
                self.bottleneck15.init_from_list(sublist)
                self.bottleneck15.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[16]
                sublist = w_pretrained[start: stop]
                self.bottleneck16.init_from_list(sublist)
                self.bottleneck16.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[17]
                sublist = w_pretrained[start: stop]
                self.bottleneck17.init_from_list(sublist)
                self.bottleneck17.enable_grad(requires_grad)

                start = stop
                stop = start + self.pars_count_per_layer[18]
                sublist = w_pretrained[start: stop]
                self.conv18.init_from_list(sublist)
                self.conv18.enable_grad(requires_grad)
        pass

    def forward(self, x):
        x = self.conv0(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)
        x = self.bottleneck8(x)
        x = self.bottleneck9(x)
        x = self.bottleneck10(x)
        x = self.bottleneck11(x)
        x = self.bottleneck12(x)
        x = self.bottleneck13(x)
        x = self.bottleneck14(x)
        x = self.bottleneck15(x)
        x = self.bottleneck16(x)
        x = self.bottleneck17(x)
        x = self.conv18(x)
        return x

