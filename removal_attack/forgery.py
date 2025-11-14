import copy
import torch.optim
from utils.parameters import load_args
from utils.utils import *
from utils.datasets import WMDataset

args = load_args()
save_file = "test"
init_state = torch.load('../result/'+ save_file +'.pth', weights_only=False)
model = get_model(args).to(args.device)
model.load_state_dict(init_state)
T_model = create_T_model(model, args.model).to(args.device)

wmDataset = WMDataset(image_dir='../data/logo10/', dim=args.num_classes, transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ]))
wmDataloader = DataLoader(wmDataset, batch_size=1, shuffle=False)
wm_o, key_o, seed_o = next(iter(wmDataloader))
wm_o, key_o = wm_o.to(args.device), key_o.to(args.device)

attackedDataset = WMDataset(image_dir='./data/other/img1/', dim=args.num_classes, transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ]))
attackedDataloader = DataLoader(attackedDataset, batch_size=1, shuffle=False)
wm_a, key_a, _ = next(iter(attackedDataloader))
wm_a, key_a = wm_a.to(dtype=torch.float32).to(args.device), key_a.to(dtype=torch.float32).to(args.device)
bn = load_bn_stats_hdf5(seed_o[0], args.device, './result/forgery/bn/'+ save_file +'.h5')

num_suc = 300
num_o_1 = 0
num_o_3 = 0
num_o_5 = 0
num_o_7 = 0
num_o_9 = 0
num_a_1 = 0
num_a_3 = 0
num_a_5 = 0
num_a_7 = 0
num_a_9 = 0

for test_num in range(num_suc):
    vo_fake = torch.randn(1, args.num_classes, requires_grad=True, device=args.device)
    va_fake = torch.randn(1, args.num_classes, requires_grad=True, device=args.device)

    s = cosine_similarity(vo_fake, key_o)
    optimizer_o = torch.optim.SGD([vo_fake], lr=0.001, momentum=0.9)
    optimizer_a = torch.optim.SGD([va_fake], lr=0.001, momentum=0.9)
    losses_o = []
    losses_a = []

    steps = 1000
    for step in range(steps):
        optimizer_o.zero_grad()
        img_out = T_model(vo_fake)
        ssim_v = ssim(img_out, wm_o)
        loss = 1 - ssim_v
        loss.backward()
        optimizer_o.step()
        losses_o.append(loss.item())
        if step == steps - 1 :
            if ssim_v > 0.9:
                num_o_9 += 1
            if ssim_v > 0.7:
                num_o_7 += 1
            if ssim_v > 0.5:
                num_o_5 += 1
            if ssim_v > 0.3:
                num_o_3 += 1
            if ssim_v > 0.1:
                num_o_1 += 1

    for step in range(steps):
        optimizer_a.zero_grad()
        img_out = T_model(va_fake)
        ssim_v = ssim(img_out, wm_a)
        loss = 1 - ssim_v
        loss.backward()
        optimizer_a.step()
        losses_a.append(loss.item())


        if step == steps - 1:
            if ssim_v > 0.9:
                num_a_9 += 1
            if ssim_v > 0.7:
                num_a_7 += 1
            if ssim_v > 0.5:
                num_a_5 += 1
            if ssim_v > 0.3:
                num_a_3 += 1
            if ssim_v > 0.1:
                num_a_1 += 1


rate_o_1 = num_o_1 / num_suc
rate_o_3 = num_o_3 / num_suc
rate_o_5 = num_o_5 / num_suc
rate_o_7 = num_o_7 / num_suc
rate_o_9 = num_o_9 / num_suc
rate_a_1 = num_a_1 / num_suc
rate_a_3 = num_a_3 / num_suc
rate_a_5 = num_a_5 / num_suc
rate_a_7 = num_a_7 / num_suc
rate_a_9 = num_a_9 / num_suc
print(f"{save_file}-target: num:{num_suc:.3f}, 0.1: {rate_o_1:.3f}, 0.3: {rate_o_3:.3f}, 0.5:{rate_o_5:.3f}, 0.7: {rate_o_7:.3f}, 0.9: {rate_o_9:.3f}")
print(f"{save_file}-untarget: 0.1: {rate_a_1:.3f}, 0.3: {rate_a_3:.3f}, 0.5:{rate_a_5:.3f}, 0.7: {rate_a_7:.3f}, 0.9: {rate_a_9:.3f}")
