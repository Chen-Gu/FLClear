# -*- coding: UTF-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

'''class TransposedLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, in_features: int, out_features: int):
        """
        构造转置线性层，与原始线性层共享权重。
        Args:
            original_linear (nn.Linear): 原始线性层。
        """
        super(TransposedLinear, self).__init__()
        # 引用原始层的权重和偏置
        self.weight = original_linear.weight  # 权重共享
        self.share_bias = original_linear.bias      # 偏置共享
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, y):
        """
        计算转置线性层的输出，使用 x = (y - b) * w 的公式。
        Args:
            y: 输入到转置线性层的特征向量。
        Returns:
            x: 转置线性层的输出。
        """
        if self.share_bias is not None:
            # 减去偏置
            y = y - self.share_bias
        # 矩阵乘法，使用共享的转置权重
        x = F.linear(y, self.weight.t())
        return x'''
class TransposedPooling(nn.Module):
    def __init__(self, mode='bilinear', scale_factor=2):
        super(TransposedPooling, self).__init__()
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x):
        upsampled = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return upsampled
# class TransposedBatchNorm(nn.Module):
#     def __init__(self, BatchNorm: nn.BatchNorm2d, num_features):
#         super(TransposedBatchNorm, self).__init__()
#         self.num_features = num_features
#         self.weight = BatchNorm.weight
#         self.bias = BatchNorm.bias
#         self.eps = BatchNorm.eps
#
#     def forward(self, y):
#         mean = y.mean(dim=(0, 2, 3), keepdim=True)
#         var = y.var(dim=(0, 2, 3), keepdim=True, unbiased=True)
#         mean = mean.view(1, self.num_features, 1, 1)
#         var = var.view(1, self.num_features, 1, 1)
#         x_normalized = (y - mean) / torch.sqrt(var + self.eps)
#         x = self.weight.view(1, self.num_features, 1, 1) * x_normalized + self.bias.view(1, self.num_features, 1, 1)
#         return x


# class TransposedBatchNorm(nn.Module):
#     def __init__(self, BatchNorm: nn.BatchNorm2d, num_features):
#         super(TransposedBatchNorm, self).__init__()
#         self.batch_norm = BatchNorm  # Store the BatchNorm module itself
#         self.register_parameter('weight', BatchNorm.weight)
#         self.register_parameter('bias', BatchNorm.bias)
#         self.num_features = num_features
#         self.eps = BatchNorm.eps
#         self.momentum = BatchNorm.momentum
#
#     def forward(self, y, mean=None, var=None):
#         if self.training:
#             mean = y.mean(dim=(0, 2, 3), keepdim=True)
#             var = y.var(dim=(0, 2, 3), keepdim=True, unbiased=True)
#             with torch.no_grad():
#                 # 使用 copy_ 替代原地操作
#                 self.batch_norm.running_mean.copy_(
#                     (1 - self.momentum) * self.batch_norm.running_mean + self.momentum * mean.squeeze()
#                 )
#                 self.batch_norm.running_var.copy_(
#                     (1 - self.momentum) * self.batch_norm.running_var + self.momentum * var.squeeze()
#                 )
#         else:
#             if mean is None and var is None:
#                 mean = self.batch_norm.running_mean.view(1, self.num_features, 1, 1)
#                 var = self.batch_norm.running_var.view(1, self.num_features, 1, 1)
#             else:
#                 mean = mean.view(1, self.num_features, 1, 1)
#                 var = var.view(1, self.num_features, 1, 1)
#
#         x_normalized = (y - mean) / torch.sqrt(var + self.eps)
#         x = self.batch_norm.weight.view(1, self.num_features, 1, 1) * x_normalized + \
#             self.batch_norm.bias.view(1, self.num_features, 1, 1)
#         return x

# class CNN(nn.Module):
#     def __init__(self, args):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(args.num_channels, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(64 * 8 * 8, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, args.num_classes)
#
#     def forward(self, x):     #(64, 3, 32, 32)
#         x = self.pool(F.relu(self.conv1(x)))    #(64, 64, 16, 16)
#         x = self.pool(F.relu(self.conv2(x)))    #(64, 128, 8, 8)
#         x = x.view(-1, 64 * 8 * 8)    #(64, 64*8*8)
#         x = F.relu(self.fc1(x))    #(64, 512)
#         x = F.relu(self.fc2(x))    #(64, 256)
#         x = self.fc3(x)    #(64, 10)
#         return x
# class TransposedCNN(nn.Module):
#     def __init__(self, CNN):
#         super(TransposedCNN, self).__init__()
#
#         self.fc3_t = TransposedLinear(CNN.fc3, 10, 256)
#         self.fc2_t = TransposedLinear(CNN.fc2,256, 512)
#         self.fc1_t = TransposedLinear(CNN.fc1,512, 64*8*8)
#         self.pool_t = TransposedPooling(CNN.pool, mode='bilinear', scale_factor=2)
#         self.conv2_t = TransposedConv(CNN.conv2)
#         self.conv1_t = TransposedConv(CNN.conv1)
#
#
#     def forward(self, x):    #(10, 10)
#         x = F.relu(self.fc3_t(x))    #(10, 256)
#         x = F.relu(self.fc2_t(x))    #(10, 512)
#         x = F.relu(self.fc1_t(x))    #(10, 64*8*8)
#         x = x.view(-1, 64, 8, 8)   #(10, 64, 8, 8)
#         x = self.pool_t(x)   #(10, 64, 16, 16)
#         x = F.relu(x)
#         x = self.conv2_t(x)   #(10, 32, 16, 16)
#         x = self.pool_t(x) #(10, 32, 32, 32)
#         x = F.relu(x)
#         x = self.conv1_t(x)     #(10, 3, 32, 32)
#         return x

# class TextEncoder(nn.Module):
#     def __init__(self, clip_model, output_dim=10):  #output_dim文本特征输出维度
#         super(TextEncoder, self).__init__()
#         self.clip = clip_model
#         self.fc = nn.Linear(512, output_dim)  # 将CLIP输出映射到目标维度,CLIP ViT-B/32 模型的文本特征输出维度是 512
#
#     def forward(self, text):
#         inputs = clip_processor(text=text, return_tensors="pt", padding=True)
#         with torch.no_grad():
#             text_features = self.clip.get_text_features(**inputs)
#         return self.fc(text_features)
class TransposedLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, in_features: int, out_features: int):
        super(TransposedLinear, self).__init__()
        # 引用原始层的权重和偏置
        self.weight = original_linear.weight  # 权重共享
        self.bias = original_linear.bias      # 偏置共享

    def forward(self, y):
        if self.bias is not None:
            y = y - self.bias
        x = F.linear(y, self.weight.t())
        return x
class TransposedConv(nn.Module):
    def __init__(self, conv_layer, stride=1, padding=1, outpadding=0):
        super().__init__()
        self.conv_layer = conv_layer
        # 保存转置卷积的超参数
        self.stride = stride
        self.padding = padding
        self.output_padding = outpadding
        # 直接引用 conv_layer 的权重（不创建新的 nn.Parameter）
        self.weight = conv_layer.weight  # 形状: [out_channels, in_channels, k, k]

    def forward(self, x):
        # 使用 F.conv_transpose2d，手动应用转置卷积
        # 输入 x: [batch, out_channels, h_in, w_in]
        # weight: [out_channels, in_channels, k, k]
        # 输出: [batch, in_channels, h_out, w_out]
        return F.conv_transpose2d(
            x,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding
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

        self.memory = dict()

    def forward(self, x):
        x = self.model1(x)
        x = x.view(x.size(0), -1)
        x = self.model2(x)
        return x

    def load_global_model(self, state_dict, device, watermark=False):
        if watermark:
            # self.memory = dict()
            for key in state_dict:
                old_weights = self.state_dict()[key]
                new_weights = state_dict[key]
                if key in self.memory:
                    self.memory[key] = torch.add(self.memory[key], torch.sub(new_weights, old_weights).to(device))
                else:
                    self.memory[key] = torch.sub(new_weights, old_weights).to(device)
        self.load_state_dict(state_dict)
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
            ('fc1', TransposedLinear(VGG13.model2.fc1, 1028, 512 * 4 * 4))
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
            ('relu3', nn.ReLU()),

            ('fc2', nn.Linear(512, 128)),
            ('relu3', nn.ReLU()),
            ('fc3', nn.Linear(128, self.num_classes))
        ]))
        self.memory = dict()

    def forward(self, x):
        x = self.model1(x)
        x = x.view(x.size(0), -1)
        x = self.model2(x)
        return x

    def load_global_model(self, state_dict, device, watermark=False):
        if watermark:
            for key in state_dict:
                old_weights = self.state_dict()[key]
                new_weights = state_dict[key]
                if key in self.memory:
                    self.memory[key] = torch.add(self.memory[key], torch.sub(new_weights, old_weights).to(device))
                else:
                    self.memory[key] = torch.sub(new_weights, old_weights).to(device)
        self.load_state_dict(state_dict)
class TransposedAlexNet(nn.Module):
    def __init__(self, AlexNet):
        super(TransposedAlexNet, self).__init__()
        self.model1 = nn.Sequential(OrderedDict([
            # ('bn5', TransposedBatchNorm(AlexNet.model1.bn5, 256)),
            ('conv5', TransposedConv(AlexNet.model1.conv5,2,1,1)),
            ('bn4', TransposedBatchNorm(AlexNet.model1.bn4, 384)),
            ('relu4', nn.ReLU()),

            ('conv4', TransposedConv(AlexNet.model1.conv4)),
            ('bn3', TransposedBatchNorm(AlexNet.model1.bn3, 384)),
            ('relu3', nn.ReLU()),

            ('conv3', TransposedConv(AlexNet.model1.conv3)),
            ('bn2',TransposedBatchNorm(AlexNet.model1.bn2,192)),
            ('relu2', nn.ReLU()),

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
    # expansion = 1
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

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.final = nn.Sequential(OrderedDict([
            # ('drop', nn.Dropout(p=0.5)),
            ('fc1', nn.Linear(512 * 4 * 4, 1024)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(1024, 512)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(512, self.num_classes))
        ]))
        self.memory = dict()
    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels :
            downsample = nn.Sequential(     #对齐数据大小和通道数，方便与输出相加
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
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.final(x)
        return x

    def load_global_model(self, state_dict, device, watermark=False):
        if watermark:
            for key in state_dict:
                old_weights = self.state_dict()[key]
                new_weights = state_dict[key]
                if key in self.memory:
                    self.memory[key] = torch.add(self.memory[key], torch.sub(new_weights, old_weights).to(device))
                else:
                    self.memory[key] = torch.sub(new_weights, old_weights).to(device)
        self.load_state_dict(state_dict)

class TransposedResNet18(nn.Module):
    def __init__(self, ResNet18):
        super(TransposedResNet18, self).__init__()
        self.in_channels = 64
        # Initial convolutional layer
        self.initial = nn.Sequential(OrderedDict([
            # ('bn1', TransposedBatchNorm(ResNet18.initial.bn1, 64)),

            ('conv1', TransposedConv(ResNet18.initial.conv1)),
            ('relu1', nn.ReLU(inplace=True)),

        ]))

        # Residual layers
        self.layer1 = self.make_layer(ResNet18.layer1, 64)
        self.layer2 = self.make_layer(ResNet18.layer2, 128,2,1,1)
        self.layer3 = self.make_layer(ResNet18.layer3, 256,2,1,1)
        self.layer4 = nn.Sequential(
            nn.Sequential(OrderedDict([
                # ('pool1', TransposedPooling()),
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
            # ('fc', TransposedLinear(ResNet18.final.fc, ResNet18.num_classes, 512)),
            ('fc3', TransposedLinear(ResNet18.final.fc3, ResNet18.num_classes, 512)),
            ('relu3', nn.ReLU()),
            ('fc2', TransposedLinear(ResNet18.final.fc2, 512, 1024)),  # 2304 * 2
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
                        # 处理嵌套的 nn.Sequential
                        for sub_name, sub_module in module._modules.items():
                            full_name = f"{layer_name}.{name}.{sub_name}"
                            if isinstance(sub_module, TransposedBatchNorm) and full_name in bn_stats:
                                x = sub_module(x, mean=bn_stats[full_name]['running_mean'],
                                               var=bn_stats[full_name]['running_var'])
                            else:
                                x = sub_module(x)
                    else:
                        # 处理非 Sequential 模块
                        full_name = f"{layer_name}.{name}"
                        if isinstance(module, TransposedBatchNorm) and full_name in bn_stats:
                            x = module(x, mean=bn_stats[full_name]['running_mean'].to(x.device),
                                       var=bn_stats[full_name]['running_mean'].to(x.device))
                        else:
                            x = module(x)
        sigmoid = nn.Sigmoid()
        return sigmoid(x)
