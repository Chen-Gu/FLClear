import copy
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.datasets import WMDataset
from utils.models import *
from utils.parameters import load_args
from utils.utils import evaluate, imshow, load_bn_stats_hdf5, ssim, wm_extract
import os
import sys
import lpips

def topk_pruning(model, k_percent):
    param_abs = []
    param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_abs.append(param.abs().detach().view(-1))
            param_names.append(name)

    all_params = torch.cat(param_abs)

    pruned_state_dict = OrderedDict()
    for name, param in model.state_dict().items():
        pruned_state_dict[name] = param.clone().detach() 
    if k_percent == 0:
        return pruned_state_dict

    k = int(len(all_params) * (k_percent / 100.0))
    threshold = torch.topk(all_params, k, largest=False)[0][-1]

    for name, param in model.named_parameters():
        if param.requires_grad:
            mask = param.abs().detach() > threshold
            pruned_param = param.clone().detach() * mask.float()
            pruned_state_dict[name] = pruned_param
        else:
            pruned_state_dict[name] = param

    return pruned_state_dict

args = load_args()
wmDataset = WMDataset(image_dir='../data/logo10/', dim = args.num_classes, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            ]))
save_file = "test"
init_model = MobileNetV2()
init_state = torch.load('../result/'+ save_file +'.pth', weights_only=False)
init_model.load_state_dict(init_state)

model = MobileNetV2()
model.load_state_dict(init_state)
T_model =  TransposedMobileNetV2(model)

for i in [0,20.0,40.0,60.0,80.0]:
    pruned_model_state = topk_pruning(init_model, k_percent=i)
    model.load_state_dict(pruned_model_state)
    test_acc, test_ave_loss = evaluate(model, args)
    build_wms, avg_ssim = wm_extract(T_model, wmDataset, args, bn_path='../result/'+ save_file +'.h5')
    print(f"ratio:{i}%, test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f},SSIM:{avg_ssim:.2f}")

