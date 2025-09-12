import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.datasets import WMDataset
from utils.models import *
from utils.parameters import load_args
from utils.utils import evaluate, imshow, load_bn_stats_hdf5, ssim, wm_extract


def quantize_tensor(tensor, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.

    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale

    # 模拟量化
    q_tensor = ((tensor / scale) + zero_point).round().clamp(qmin, qmax)
    # 反量化（转换回float）
    return (q_tensor - zero_point) * scale

def apply_weight_quantization_attack(model, bit_num):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                module.weight.copy_(quantize_tensor(module.weight, bit_num))
                if module.bias is not None:
                    module.bias.copy_(quantize_tensor(module.bias, bit_num))


if __name__ == "__main__":
    # 创建模型并初始化
    args = load_args()
    wmDataset = WMDataset(image_dir='./data/logo10/', dim=args.num_classes, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]))

    init_state = torch.load('./result/model/AlexNet_C10_FedAvg_wm.pth', weights_only=False)
    model = AlexNet()
    model.load_state_dict(init_state)
    T_model =  TransposedAlexNet(model)

    acc = []
    SSIM = []
    test_acc, test_ave_loss = evaluate(model, args)

    build_wms, avg_ssim = wm_extract(T_model, wmDataset, args, bn_path='./result/bn/AlexNet_C10_FedAvg_wm.h5')
    print(f"epoch:{0}, test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f}, SSIM:{avg_ssim:.2f}")

    bit_list = [16,8,4,2]
    for bit in bit_list:
        apply_weight_quantization_attack(model, bit)
        test_acc, test_ave_loss = evaluate(model, args)
        acc.append(test_acc)

        build_wms, avg_ssim = wm_extract(T_model, wmDataset, args, bn_path='./result/bn/AlexNet_C10_FedAvg_wm.h5')

        # plt.figure(figsize=(6, 1))
        # for j in range(build_wms.shape[0]):
        #     plt.subplot(1, build_wms.shape[0], j + 1)
        #     imshow(build_wms[j])
        #     if j == 0:
        #         plt.title(f"Quantization accuracy: {bit}")
        # plt.show()

        print(f"bit :{bit} test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f},SSIM:{avg_ssim:.2f}")
        SSIM.append(avg_ssim * 100)

    # plt.figure(figsize=(8, 6))
    # x = bit_list
    # plt.plot(x, acc, marker='o', linestyle='-', color='b', label='test_acc')
    # plt.plot(x, SSIM, marker='o', linestyle='-', color='g', label='watermark')
    # plt.title('Quantization')
    # plt.xlabel('Quantization Precision (bits)')
    # plt.ylabel('result')
    # plt.grid(True)
    # plt.legend()
    # plt.show()
