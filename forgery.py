
import torch.optim
from utils.parameters import load_args
from utils.utils import *
from utils.datasets import WMDataset

args = load_args()

init_state = torch.load('./result/model/AlexNet_C10_FedAvg_wm.pth', weights_only=False)
model = AlexNet().to(args.device)
model.load_state_dict(init_state)
T_model =  TransposedAlexNet(model).to(args.device)

wmDataset = WMDataset(image_dir='./data/logo10/', dim=args.num_classes, transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ]))
wmDataloader = DataLoader(wmDataset, batch_size=1, shuffle=False)
wm_o, key_o, seed_o = next(iter(wmDataloader))
wm_o, key_o = wm_o.to(args.device), key_o.to(args.device)

attackedDataset = WMDataset(image_dir='./data/visual_image/', dim=args.num_classes, transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ]))
attackedDataloader = DataLoader(attackedDataset, batch_size=1, shuffle=False)
wm_a, key_a, _ = next(iter(attackedDataloader))
wm_a, key_a = wm_a.to(dtype=torch.float32).to(args.device), key_a.to(dtype=torch.float32).to(args.device)
bn = load_bn_stats_hdf5(seed_o[0], args.device, './result/bn/AlexNet_C10_FedAvg_wm.h5')

num_suc_o = 0
num_suc_a = 0
num_suc = 100
for test_num in range(num_suc):
    vo_fake = torch.randn(1, args.num_classes, requires_grad=True, device=args.device)
    va_fake = torch.randn(1, args.num_classes, requires_grad=True, device=args.device)

    s = cosine_similarity(vo_fake, key_o)
    # print(s)
    optimizer_o = torch.optim.SGD([vo_fake], lr=0.001, momentum=0.9)
    optimizer_a = torch.optim.SGD([va_fake], lr=0.001, momentum=0.9)
    losses_o = []
    losses_a = []

    # 定向伪造图像
    steps = 1000
    for step in range(steps):
        optimizer_o.zero_grad()
        img_out = T_model(vo_fake)
        ssim_v = ssim(img_out, wm_o)
        loss = 1 - ssim_v
        loss.backward()
        optimizer_o.step()
        losses_o.append(loss.item())
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1} | o - Loss: {loss.item():.4f}, ssim:{ssim_v:.4f}")
        if step == steps - 1 and loss <= 0.5:
            print("successful")
            num_suc_o += 1
    # 显示定向伪造图像
    # if step == steps - 1:
    #     T_model.eval()
    #     with torch.no_grad():
    #         rewm = T_model(key_o)
    #     plt.subplot(1, 2, 1)
    #     imshow(wm_o[0])
    #     plt.title("our wm")
    #     plt.subplot(1, 2, 2)
    #     imshow(img_out[0])
    #     plt.title("attacker re_wm")
    #     plt.savefig(f"./result/img/AlexNet_o_ssim_{ssim_v:0.2f}_{test_num}.png")
    #     plt.close()

    for step in range(steps):
        optimizer_a.zero_grad()
        img_out = T_model(va_fake)
        ssim_v = ssim(img_out, wm_a)
        loss = 1 - ssim_v
        loss.backward()
        optimizer_a.step()
        losses_a.append(loss.item())

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1} | a - Loss: {loss.item():.4f}, ssim:{ssim_v:.4f}")
        if step == steps - 1 and loss <= 0.5:
            print("successful")
            num_suc_a += 1
    # 显示伪造图像
    # if step == steps - 1:
    #         plt.subplot(1, 2, 1)
    #         imshow(wm_a[0])
    #         plt.title("goal wm")
    #         plt.subplot(1, 2, 2)
    #         imshow(img_out[0])
    #         plt.title("attacker re_wm")
    #         plt.savefig(f"./result/img/AlexNet_a_ssim_{ssim_v:0.2f}_{test_num}.png")
    #         plt.close()
rate_o = num_suc_o / num_suc
rate_a = num_suc_a / num_suc
print(f"num:{num_suc}, num_suc_o: {num_suc_o}, num_suc_a: {num_suc_a}")
print(f"rate_o: {rate_o}, rate_a: {rate_a}")
# epochs = [i for i in range(steps)]
# plt.plot(epochs, losses_o, label='Targeted forgery attack')
# plt.plot(epochs, losses_a, label='Untargeted forgery attack')
# plt.title(f"The loss of forgery attack")
# plt.legend()
# plt.grid(True)
# plt.savefig(f"./result/img/AlexNet_loss.png")
# plt.close()