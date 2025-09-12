import copy
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
from utils.datasets import WMDataset
from utils.models import *
from utils.parameters import load_args
from utils.utils import evaluate, imshow, load_bn_stats_hdf5, ssim, wm_extract
import os
import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
sys.path.append('/data/home/wls_syy/code/FedTracker')
def topk_pruning(model, k_percent):
    """
    对模型的所有参数进行 top-k% 剪枝，基于参数绝对值大小。
    Args:
        model: PyTorch 模型 (nn.Module)
        k_percent: 剪枝百分比（0到100之间的浮点数）
    Returns:
    """
    param_abs = []
    param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_abs.append(param.abs().detach().view(-1))
            param_names.append(name)

    # 合并所有参数到一个向量
    all_params = torch.cat(param_abs)

    pruned_state_dict = OrderedDict()
    # 获取原始模型的完整状态字典（包括参数和缓冲区）
    for name, param in model.state_dict().items():
        pruned_state_dict[name] = param.clone().detach()  # 深拷贝张量
    if k_percent == 0:
        return pruned_state_dict

    # 计算 top-k% 阈值
    k = int(len(all_params) * (k_percent / 100.0))
    threshold = torch.topk(all_params, k, largest=False)[0][-1]

    # 对每个参数应用剪枝
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 创建掩码：保留绝对值大于阈值的参数
            mask = param.abs().detach() > threshold
            pruned_param = param.clone().detach() * mask.float()
            pruned_state_dict[name] = pruned_param
        else:
            pruned_state_dict[name] = param

    return pruned_state_dict

# if __name__ == "__main__":
    # 创建模型并初始化

args = load_args()
wmDataset = WMDataset(image_dir='./data/logo10/', dim = args.num_classes, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            ]))

init_model =ResNet18()
init_state = torch.load('./result/model/ResNet_C10_FedAvg_wm.pth', weights_only=False)
init_model.load_state_dict(init_state)

model = ResNet18()
model.load_state_dict(init_state)
T_model =  TransposedResNet18(model)

acc = []
SSIM = []

for i in [0, 30.0, 50.0, 70.0, 90.0]:
    pruned_model_state = topk_pruning(init_model, k_percent=i)
    model.load_state_dict(pruned_model_state)
    test_acc, test_ave_loss = evaluate(model, args)
    acc.append(test_acc)

    build_wms, avg_ssim = wm_extract(T_model, wmDataset, args, bn_path='./result/bn/ResNet_C10_FedAvg_wm.h5')

    plt.figure(figsize=(6, 1))
    for j in range(build_wms.shape[0]):
        plt.subplot(1, build_wms.shape[0], j + 1)
        imshow(build_wms[j])
        if j == 0:
            plt.title(f"prune ratio: {i}%")
    plt.show()

    print(f"test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f},SSIM:{avg_ssim:.2f}")
    SSIM.append(avg_ssim * 100)

# print(SSIM)
# plt.figure(figsize=(8, 6))
# x = np.arange(len(acc)) * 0.1
# plt.plot(x, acc, marker='o', linestyle='-', color='b', label='test_acc')
# plt.plot(x, SSIM, marker='o', linestyle='-', color='g', label='watermark')
# plt.title('pruning test')  # 设置标题
# plt.xlabel('ratio')  # 设置 x 轴标签
# plt.ylabel('result')  # 设置 y 轴标签
# plt.grid(True)  # 添加网格线
# plt.legend()  # 添加图例
# plt.show()
