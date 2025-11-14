import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.datasets import WMDataset
from utils.models import *
from utils.parameters import load_args
from utils.utils import *


def quantize_tensor(tensor, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    q_tensor = ((tensor / scale) + zero_point).round().clamp(qmin, qmax)
    return (q_tensor - zero_point) * scale

def apply_weight_quantization_attack(model, bit_num):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                module.weight.copy_(quantize_tensor(module.weight, bit_num))
                if module.bias is not None:
                    module.bias.copy_(quantize_tensor(module.bias, bit_num))


if __name__ == "__main__":
    args = load_args()
    wmDataset = WMDataset(image_dir='../data/logo10/', dim=args.num_classes, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]))
    save_file = 'test' 
    init_state = torch.load('../result/'+ save_file +'.pth', weights_only=False)
    model = ResNet18()
    model.load_state_dict(init_state)
    T_model =  TransposedResNet18(model)
    plt.figure(figsize=(6, 1))
    for j in range(build_wms.shape[0]):
        plt.subplot(1, build_wms.shape[0], j + 1)
        imshow(build_wms[j])
        if j == 0:
            plt.title(f"Quantization accuracy: 0")
    plt.savefig(f'./result/t/attack/{save_file}_quan0.png')
    bit_list = [16,8,4,2]
    for bit in bit_list:
        apply_weight_quantization_attack(model, bit)
        test_acc, test_ave_loss = evaluate(model, args)
        acc.append(test_acc)

        build_wms, avg_ssim = wm_extract(T_model, wmDataset, args, bn_path='../result/'+ save_file +'.h5')

        plt.figure(figsize=(6, 1))
        for j in range(build_wms.shape[0]):
            plt.subplot(1, build_wms.shape[0], j + 1)
            imshow(build_wms[j])
            if j == 0:
                plt.title(f"Quantization accuracy: {bit}")
        plt.savefig(f'./result//{save_file}_quan{bit}.png')

        print(f"bit :{bit} test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f},SSIM:{avg_ssim:.2f}")
 
