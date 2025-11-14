# -*- coding: UTF-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class TransposedLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, in_features: int, out_features: int):
        super(TransposedLinear, self).__init__()

        self.weight = original_linear.weight
        self.bias = original_linear.bias

    def forward(self, y):
        if self.bias is not None:
            y = y - self.bias
        x = F.linear(y, self.weight.t())
        return x

class TransposedConv(nn.Module):
    def __init__(self, conv_layer, stride=1, padding=1, outpadding=0):
        super().__init__()
        self.conv_layer = conv_layer

        self.stride = stride
        self.padding = padding
        self.output_padding = outpadding
        self.groups = conv_layer.groups

        self.weight = conv_layer.weight

    def forward(self, x):
        return F.conv_transpose2d(
            x,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
        )

class TransposedBatchNorm(nn.Module):
    def __init__(self, BatchNorm: nn.BatchNorm2d, num_features):
        super(TransposedBatchNorm, self).__init__()
        self.num_features = num_features
        self.register_parameter('weight', BatchNorm.weight)
        self.register_parameter('bias', BatchNorm.bias)
        self.eps = BatchNorm.eps
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.momentum = BatchNorm.momentum

    def forward(self, y, mean=None, var=None):
        if self.training:
            mean = y.mean(dim=(0, 2, 3), keepdim=True)
            var = y.var(dim=(0, 2, 3), keepdim=True, unbiased=True)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.squeeze())
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.squeeze())
        else:
            if mean is None and var is None:
                mean = self.running_mean.view(1, self.num_features, 1, 1)
                var = self.running_var.view(1, self.num_features, 1, 1)
            else:
                mean = mean.view(1, self.num_features, 1, 1)
                var = var.view(1, self.num_features, 1, 1)
        x_normalized = (y - mean) / torch.sqrt(var + self.eps)
        x = self.weight.view(1, self.num_features, 1, 1) * x_normalized + self.bias.view(1, self.num_features, 1, 1)
        return x

class VGG13(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.num_classes = num_classes
        self.model1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d((2, 2), (2, 2))),

            ('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('relu4', nn.ReLU(inplace=True)),

            # ('dropout1', nn.Dropout(p=0.25)),

            ('conv5', nn.Conv2d(128, 256, 3, padding=1, bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv6', nn.Conv2d(256, 256, 3, padding=1, bias=False)),
            ('bn6', nn.BatchNorm2d(256)),
            ('relu6', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d((2, 2), (2, 2))),

            ('conv7', nn.Conv2d(256, 512, 3, padding=1, bias=False)),
            ('bn7', nn.BatchNorm2d(512)),
            ('relu7', nn.ReLU(inplace=True)),
            ('conv8', nn.Conv2d(512, 512, 3, padding=1, bias=False)),
            ('bn8', nn.BatchNorm2d(512)),
            ('relu8', nn.ReLU(inplace=True)),

            ('conv9', nn.Conv2d(512, 512, 3, padding=1, bias=False)),
            ('bn9', nn.BatchNorm2d(512)),
            ('relu9', nn.ReLU(inplace=True)),

            # ('dropout2', nn.Dropout(p=0.25)),

            ('conv10', nn.Conv2d(512, 512, 3, padding=1, bias=False)),
            ('bn10', nn.BatchNorm2d(512)),
            ('relu10', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d((2, 2), (2, 2))),

        ]))
        self.model2 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512 * 4 * 4, 1028)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(1028, 128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(128, self.num_classes))
        ]))

    def forward(self, x):
        x = self.model1(x)
        x = x.view(x.size(0), -1)
        x = self.model2(x)
        return x

class TransposedVGG13(nn.Module):
    def __init__(self, VGG13):
        super(TransposedVGG13, self).__init__()
        self.model1 = nn.Sequential(OrderedDict([
            # ('bn10', TransposedBatchNorm(VGG13.model1.bn10, num_features=512)),
            ('conv10', TransposedConv(VGG13.model1.conv10,2,1,1)),
            ('bn9', TransposedBatchNorm(VGG13.model1.bn9, num_features=512)),
            ('relu9', nn.ReLU(inplace=True)),
            # ('pool1', TransposedPooling()),

            ('conv9', TransposedConv(VGG13.model1.conv9)),
            ('bn8', TransposedBatchNorm(VGG13.model1.bn8, num_features=512)),
            ('relu8', nn.ReLU(inplace=True)),

            ('conv8', TransposedConv(VGG13.model1.conv8)),
            ('bn7', TransposedBatchNorm(VGG13.model1.bn7, num_features=512)),
            ('relu7', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout(0.2)),
            ('conv7', TransposedConv(VGG13.model1.conv7)),
            ('bn6', TransposedBatchNorm(VGG13.model1.bn6, num_features=256)),
            ('relu6', nn.ReLU(inplace=True)),

            ('conv6', TransposedConv(VGG13.model1.conv6,2,1,1)),  # [10,512,4,4]
            ('bn5', TransposedBatchNorm(VGG13.model1.bn5, num_features=256)),
            ('relu5', nn.ReLU(inplace=True)),
            # ('pool2', TransposedPooling()),

            ('conv5', TransposedConv(VGG13.model1.conv5)),
            ('bn4', TransposedBatchNorm(VGG13.model1.bn4, num_features=128)),
            ('relu4', nn.ReLU(inplace=True)),

            ('conv4', TransposedConv(VGG13.model1.conv4)),  # [10,512,4,4]
            ('bn3', TransposedBatchNorm(VGG13.model1.bn3, num_features=128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('dropout', nn.Dropout(0.2)),
            ('conv3', TransposedConv(VGG13.model1.conv3)),
            ('bn2', TransposedBatchNorm(VGG13.model1.bn2, num_features=64)),
            ('relu2', nn.ReLU(inplace=True)),

            ('conv2', TransposedConv(VGG13.model1.conv2,2,1,1)),
            ('bn1', TransposedBatchNorm(VGG13.model1.bn1, num_features=64)),
            ('relu1', nn.ReLU(inplace=True)),
            # ('pool3', TransposedPooling()),

            ('conv1', TransposedConv(VGG13.model1.conv1)),
        ]))
        self.model2 = nn.Sequential(OrderedDict([
            ('fc3', TransposedLinear(VGG13.model2.fc3, VGG13.num_classes, 128)),
            ('relu', nn.ReLU(inplace=True)),
            ('fc2', TransposedLinear(VGG13.model2.fc2, 128, 1028)),  # 2304 * 2
            ('relu', nn.ReLU(inplace=True)),
            ('fc1', TransposedLinear(VGG13.model2.fc1, 1028, 512 * 4 * 4)),
            ('dropout', nn.Dropout(0.2)),
        ]))
    def forward(self, x, bn_stats=None):
        x = self.model2(x)
        x = x.view(-1, 512, 4, 4)
        if not bn_stats:
            x = self.model1(x)
        else:
            for name, module in self.model1._modules.items():
                if isinstance(module, TransposedBatchNorm) and 'model1.' + name in bn_stats:
                    tname = 'model1.' + name
                    x = module(x, mean=bn_stats[tname]['running_mean'],
                               var=bn_stats[tname]['running_var'])
                else:
                    x = module(x)

        sigmoid = nn.Sigmoid()
        return sigmoid(x)

class AlexNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.model1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))),

            ('conv2', nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(192)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))),

            ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(384)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False)),
            ('bn4', nn.BatchNorm2d(384)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('relu5', nn.ReLU()),
            ('pool3', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))),
        ]))
        self.model2 = nn.Sequential(OrderedDict([

            ('fc1', nn.Linear(256 * 4 * 4, 512)),
            ('relu1', nn.ReLU()),

            ('fc2', nn.Linear(512, 128)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(128, self.num_classes))
        ]))

    def forward(self, x):
        x = self.model1(x)
        x = x.view(x.size(0), -1)
        x = self.model2(x)
        return x

class TransposedAlexNet(nn.Module):
    def __init__(self, AlexNet):
        super(TransposedAlexNet, self).__init__()
        self.model1 = nn.Sequential(OrderedDict([

            ('conv5', TransposedConv(AlexNet.model1.conv5,2,1,1)),
            ('bn4', TransposedBatchNorm(AlexNet.model1.bn4, 384)),
            ('relu4', nn.ReLU()),

            ('conv4', TransposedConv(AlexNet.model1.conv4)),
            ('bn3', TransposedBatchNorm(AlexNet.model1.bn3, 384)),
            ('relu3', nn.ReLU()),

            ('conv3', TransposedConv(AlexNet.model1.conv3)),
            ('bn2',TransposedBatchNorm(AlexNet.model1.bn2,192)),
            ('relu2', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('conv2', TransposedConv(AlexNet.model1.conv2,2,1,1)),
            ('bn1', TransposedBatchNorm(AlexNet.model1.bn1, num_features=64)),
            ('relu1', nn.ReLU()),

            ('conv1', TransposedConv(AlexNet.model1.conv1,2,1,1)),
        ]))
        self.model2 = nn.Sequential(OrderedDict([
            ('fc3', TransposedLinear(AlexNet.model2.fc3, AlexNet.num_classes, 128)),
            ('relu3', nn.ReLU()),

            ('fc2', TransposedLinear(AlexNet.model2.fc2, 128, 512)),  # 2304 * 2
            ('relu2', nn.ReLU()),
            ('fc1', TransposedLinear(AlexNet.model2.fc1, 512, 256 * 4 * 4)),
            ('dropout', nn.Dropout(0.2)),
        ]))
    def forward(self, x, bn_stats=None):
        x = self.model2(x)
        x = x.view(-1, 256, 4, 4)
        if not bn_stats:
            x = self.model1(x)
        else:
            for name, module in self.model1._modules.items():
                if isinstance(module, TransposedBatchNorm) and 'model1.' + name in bn_stats:
                    tname = 'model1.' + name
                    x = module(x, mean=bn_stats[tname]['running_mean'],
                               var=bn_stats[tname]['running_var'])
                else:
                    x = module(x)
        sigmoid = nn.Sigmoid()
        return sigmoid(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels))
        ]))
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# ResNet-18 Model
class ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.num_classes = num_classes
        # Initial convolutional layer
        self.initial = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))

        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)


        self.final = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(512 * 4 * 4, 1024)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(1024, 512)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(512, self.num_classes))
        ]))


    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels :
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        x = self.final(x)
        return x

class TransposedResNet18(nn.Module):
    def __init__(self, ResNet18):
        super(TransposedResNet18, self).__init__()
        self.in_channels = 64

        self.initial = nn.Sequential(OrderedDict([

            ('conv1', TransposedConv(ResNet18.initial.conv1)),
            ('relu1', nn.ReLU(inplace=True)),

        ]))

        # Residual layers
        self.layer1 = self.make_layer(ResNet18.layer1, 64)
        self.layer2 = self.make_layer(ResNet18.layer2, 128,2,1,1)
        self.layer3 = self.make_layer(ResNet18.layer3, 256,2,1,1)
        self.layer4 = nn.Sequential(
            nn.Sequential(OrderedDict([

                ('conv2', TransposedConv(ResNet18.layer4[1].block[3])),
                ('bn1', TransposedBatchNorm(ResNet18.layer4[1].block[1], 512)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv1', TransposedConv(ResNet18.layer4[1].block[0])),
            ])),
            nn.Sequential(OrderedDict([
                ('bn2', TransposedBatchNorm(ResNet18.layer4[0].block[4], 512)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv2', TransposedConv(ResNet18.layer4[0].block[3])),
                ('bn1', TransposedBatchNorm(ResNet18.layer4[0].block[1], 512)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv1', TransposedConv(ResNet18.layer4[0].block[0],2,1,1)),
                ('relu0', nn.ReLU(inplace=True)),


            ]))
        )

        self.final = nn.Sequential(OrderedDict([
            ('fc3', TransposedLinear(ResNet18.final.fc3, ResNet18.num_classes, 512)),
            ('relu3', nn.ReLU()),
            ('fc2', TransposedLinear(ResNet18.final.fc2, 512, 1024)),
            ('relu2', nn.ReLU()),
            ('fc1', TransposedLinear(ResNet18.final.fc1, 1024, 512 * 4 * 4)),
        ]))

    def make_layer(self, layers, in_channels, stride=1, padding=1, outpadding=0):
        layer = nn.Sequential(
            nn.Sequential(OrderedDict([
                ('bn2', TransposedBatchNorm(layers[1].block[4], in_channels)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv2', TransposedConv(layers[1].block[3])),
                ('bn1', TransposedBatchNorm(layers[1].block[1], in_channels)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv1', TransposedConv(layers[1].block[0])),
            ])),
            nn.Sequential(OrderedDict([
                ('bn2', TransposedBatchNorm(layers[0].block[4], in_channels)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv2', TransposedConv(layers[0].block[3])),
                ('bn1', TransposedBatchNorm(layers[0].block[1], in_channels)),
                ('relu1', nn.ReLU(inplace=True)),
                ('dropout', nn.Dropout(0.2)),
                ('conv1', TransposedConv(layers[0].block[0], stride, padding, outpadding)),

            ]))
        )
        return layer

    def forward(self, x, bn_stats=None):
        x = self.final(x)
        x = x.view(-1,512,4,4)
        if not bn_stats:
            x = self.layer4(x)
            x = self.layer3(x)
            x = self.layer2(x)
            x = self.layer1(x)
            x = self.initial(x)
        else:

            layers = [
                ('layer4', self.layer4),
                ('layer3', self.layer3),
                ('layer2', self.layer2),
                ('layer1', self.layer1),
                ('initial', self.initial)
            ]
            for layer_name, layer in layers:
                for name, module in layer._modules.items():
                    if isinstance(module, nn.Sequential):

                        for sub_name, sub_module in module._modules.items():
                            full_name = f"{layer_name}.{name}.{sub_name}"
                            if isinstance(sub_module, TransposedBatchNorm) and full_name in bn_stats:
                                x = sub_module(x, mean=bn_stats[full_name]['running_mean'],
                                               var=bn_stats[full_name]['running_var'])
                            else:
                                x = sub_module(x)
                    else:

                        full_name = f"{layer_name}.{name}"
                        if isinstance(module, TransposedBatchNorm) and full_name in bn_stats:
                            x = module(x, mean=bn_stats[full_name]['running_mean'].to(x.device),
                                       var=bn_stats[full_name]['running_mean'].to(x.device))
                        else:
                            x = module(x)
        sigmoid = nn.Sigmoid()
        return sigmoid(x)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, activation=nn.ReLU):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            activation(inplace=True) if activation is not None else nn.Identity()
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, activation=nn.ReLU))

        layers.append(
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, activation=nn.ReLU))

        layers.append(ConvBNReLU(hidden_dim, oup, kernel_size=1, activation=None))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()

        def _make_divisible(v, divisor=round_nearest):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        block = InvertedResidual
        input_channel = 32
        self.last_channel = 320
        self.final_spatial_size = 4

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 1],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(input_channel * width_mult)

        features = [ConvBNReLU(3, input_channel, kernel_size=3, stride=1, activation=nn.ReLU)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel


        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Linear(self.last_channel*self.final_spatial_size*self.final_spatial_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TransposedInvertedResidual(nn.Module):
    def __init__(self, original_block: InvertedResidual):
        super(TransposedInvertedResidual, self).__init__()

        layers_f = original_block.conv

        self.inp = layers_f[-1][0].in_channels if len(layers_f) == 2 else layers_f[0][0].in_channels
        self.oup = layers_f[-1][0].out_channels

        if len(layers_f) == 3:

            idx_exp, idx_dw, idx_proj = 0, 1, 2
        else:

            idx_dw, idx_proj = 0, 1
            idx_exp = None

        proj_conv_f = layers_f[idx_proj][0]
        proj_bn_f = layers_f[idx_proj][1]
        self.t_proj = nn.Sequential(OrderedDict([

            ('t_bn', TransposedBatchNorm(proj_bn_f, proj_conv_f.out_channels)),
            ('t_relu6', nn.ReLU(inplace=True)),
            ('t_conv', TransposedConv(proj_conv_f, stride=1, padding=0, outpadding=0)),

        ]))

        dw_conv_f = layers_f[idx_dw][0]
        dw_bn_f = layers_f[idx_dw][1]

        if dw_conv_f.stride == (2, 2) or dw_conv_f.stride == 2:
            outpadding = 1
        else:
            outpadding = 0
        self.t_dw = nn.Sequential(OrderedDict([
            ('t_bn', TransposedBatchNorm(dw_bn_f, dw_conv_f.out_channels)),
            ('t_relu6', nn.ReLU(inplace=True)),
            ('t_conv', TransposedConv(dw_conv_f, stride=dw_conv_f.stride, padding=dw_conv_f.padding,
                                      outpadding=outpadding)),
        ]))

        self.t_expansion = nn.Identity()
        if idx_exp is not None:
            # Conv: inp -> hidden。 T-Conv: hidden -> inp。
            exp_conv_f = layers_f[idx_exp][0]
            exp_bn_f = layers_f[idx_exp][1]
            self.t_expansion = nn.Sequential(OrderedDict([
                ('t_bn', TransposedBatchNorm(exp_bn_f, exp_conv_f.out_channels)),
                ('t_conv', TransposedConv(exp_conv_f, stride=1, padding=0, outpadding=0)),  # 1x1 Conv always s=1, p=0
            ]))

        self.use_res_connect = original_block.use_res_connect

    def forward(self, x):
        self.identity = x
        x = self.t_proj(x)
        x = self.t_dw(x)
        x = self.t_expansion(x)
        return x


def _run_recursive_forward(module, x, bn_stats, prefix=''):
    if isinstance(module, (nn.Sequential, nn.ModuleList, OrderedDict)):
        for name, sub_module in module.named_children():
            x = _run_recursive_forward(sub_module, x, bn_stats, prefix + name + '.')
        return x

    elif isinstance(module, TransposedBatchNorm) and bn_stats is not None:

        full_name = prefix.rstrip('.')

        if full_name in bn_stats:
            mean = bn_stats[full_name]['running_mean'].to(x.device)
            var = bn_stats[full_name]['running_var'].to(x.device)

            return module(x, mean=mean, var=var)

    elif isinstance(module, TransposedInvertedResidual):
        x = _run_recursive_forward(module.t_proj, x, bn_stats, prefix + 't_proj.')
        x = _run_recursive_forward(module.t_dw, x, bn_stats, prefix + 't_dw.')
        x = _run_recursive_forward(module.t_expansion, x, bn_stats, prefix + 't_expansion.')
        return x
    else:
        return module(x)
class TransposedMobileNetV2(nn.Module):
    def __init__(self, original_mnet: MobileNetV2, num_classes=10):
        super().__init__()

        self.last_channel = original_mnet.last_channel
        self.final_spatial_size = 4

        # Classifier: Dropout -> Linear(last_channel -> num_classes)
        self.t_head = nn.Sequential(OrderedDict([

            ('t_linear2', TransposedLinear(original_mnet.classifier[2], num_classes, 512)),
            ('t_relu', nn.ReLU(inplace=True)),
            ('t_linear1', TransposedLinear(original_mnet.classifier[0], 512,
                                          self.last_channel * self.final_spatial_size * self.final_spatial_size)),
        ]))

        t_features = []

        for i in reversed(range(1, len(original_mnet.features))):
            original_block = original_mnet.features[i]
            t_features.append(TransposedInvertedResidual(original_block))

        self.t_features = nn.Sequential(*t_features)

        f_initial_conv_bn_relu = original_mnet.features[0]
        f_initial_conv = f_initial_conv_bn_relu[0]
        f_initial_bn = f_initial_conv_bn_relu[1]

        self.t_initial = nn.Sequential(OrderedDict([
            ('t_bn', TransposedBatchNorm(f_initial_bn, f_initial_conv.out_channels)),
            ('t_relu', nn.ReLU6(inplace=True)),
            ('t_conv', TransposedConv(f_initial_conv, stride=1, padding=1, outpadding=0)),
        ]))

    def forward(self, x, bn_stats=None):
        x = self.t_head(x)
        x = x.view(-1, self.last_channel, self.final_spatial_size, self.final_spatial_size)
        if bn_stats is None:

            x = self.t_features(x)
            x = self.t_initial(x)
        else:
            x = _run_recursive_forward(self.t_features, x, bn_stats, prefix='t_features.')
            x = _run_recursive_forward(self.t_initial, x, bn_stats, prefix='t_initial.')
        return nn.Sigmoid()(x)



    def __init__(self, AlexNet1):
        super(TransposedAlexNet1, self).__init__()
        self.size = AlexNet1.size
        self.model1 = nn.Sequential(OrderedDict([

            ('conv5', TransposedConv(AlexNet1.model1.conv5,2,1,1)),
            ('bn4', TransposedBatchNorm(AlexNet1.model1.bn4, 384)),
            ('relu4', nn.ReLU()),

            ('conv4', TransposedConv(AlexNet1.model1.conv4)),
            ('bn3', TransposedBatchNorm(AlexNet1.model1.bn3, 384)),
            ('relu3', nn.ReLU()),

            ('conv3', TransposedConv(AlexNet1.model1.conv3)),
            ('bn2',TransposedBatchNorm(AlexNet1.model1.bn2,192)),
            ('relu2', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('conv2', TransposedConv(AlexNet1.model1.conv2,2,1,1)),
            ('bn1', TransposedBatchNorm(AlexNet1.model1.bn1, num_features=64)),
            ('relu1', nn.ReLU()),

            ('conv1', TransposedConv(AlexNet1.model1.conv1,2,1,1)),
        ]))
        self.model2 = nn.Sequential(OrderedDict([
            ('fc3', TransposedLinear(AlexNet1.model2.fc3, AlexNet1.num_classes, 128)),
            ('relu3', nn.ReLU()),

            ('fc2', TransposedLinear(AlexNet1.model2.fc2, 128, 512)),
            ('relu2', nn.ReLU()),
            ('fc1', TransposedLinear(AlexNet1.model2.fc1, 512, 256 * AlexNet1.size * AlexNet1.size)),
            ('dropout', nn.Dropout(0.2)),
        ]))
    def forward(self, x, bn_stats=None):
        x = self.model2(x)
        x = x.view(-1, 256, self.size, self.size)
        if not bn_stats:
            x = self.model1(x)
        else:
            for name, module in self.model1._modules.items():
                if isinstance(module, TransposedBatchNorm) and 'model1.' + name in bn_stats:
                    tname = 'model1.' + name
                    x = module(x, mean=bn_stats[tname]['running_mean'],
                               var=bn_stats[tname]['running_var'])
                else:
                    x = module(x)
        sigmoid = nn.Sigmoid()
        return sigmoid(x)