import time

import matplotlib.pyplot as plt
import torch.optim

from fed.client import compute_loss_wm
from utils.parameters import load_args
from utils.utils import *
from utils.datasets import WMDataset, VectorAugment


def WMextract(T_model, wmDataloader, args, epoch, bn_path = '../result/bn/VGG13bn_stats.h5') :
    T_model.eval()
    vectors = []
    vectors.append(torch.randn(10).unsqueeze(0).to(args.device))
    vectors.append(torch.rand(10).unsqueeze(0).to(args.device))
    vectors.append(torch.bernoulli(torch.full((10,), 0.5).unsqueeze(0).to(args.device)))
    vectors.append(torch.poisson(torch.full((10,), 2.0).unsqueeze(0).to(args.device)))
    vectors.append(torch.empty(10).exponential_(lambd=1).unsqueeze(0).to(args.device))
    laplace_dist = torch.distributions.Laplace(loc=0, scale=1)
    vec = laplace_dist.sample((10,)).unsqueeze(0).to(args.device)
    vectors.append(vec)
    vector_rand = torch.cat(vectors, dim=0)
    with torch.no_grad():
        re_wms = []
        e_ssim = []
        for num, (wm, key, label) in enumerate(wmDataloader):
            key = key.to(dtype=torch.float32).to(args.device)
            wm = wm.to(dtype=torch.float32).to(args.device)
            bn = load_bn_stats_hdf5(label[0], args.device, bn_path)
            re_wm = T_model(key, bn)
            re_wms.append(re_wm)
            ssim_v = ssim(re_wm, wm)
            e_ssim.append(ssim_v.item())

            rand_out = T_model(vector_rand, bn)
            for j in range(6):
                plt.subplot(10, 6, j + num * 6 + 1)
                imshow(rand_out[j])
        plt.show()
        re_wms = torch.cat(re_wms, dim=0)
        for j in range(10):
            plt.subplot(1, 10, j + 1)
            imshow(re_wms[j])
            if j == 0:
                plt.title("epoch:" + str(epoch + 1))
        plt.show()
    return sum(e_ssim) / len(e_ssim)


args = load_args()
attackedDataset = WMDataset(image_dir='./data/visual_image/', dim=args.num_classes, transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ]))
split_data = VectorAugment(attackedDataset, [1,], args.num_classes)
split_dataloader = DataLoader(split_data, batch_size=32, shuffle=False)
attackedDataloader = DataLoader(attackedDataset, batch_size=1, shuffle=False)

WMDataset = WMDataset(image_dir='./data/logo10/', dim=args.num_classes, transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ]))


init_state = torch.load('./result/model/AlexNet_C10_FedAvg_wm.pth', weights_only=False)
model = AlexNet().to(args.device)
model.load_state_dict(init_state)
T_model =  TransposedAlexNet(model).to(args.device)

ACC_list = []
SSIM_list = []
SSIM_ATT = []
test_acc, test_ave_loss = evaluate(model, args)
ACC_list.append(test_acc)

build_wms, avg_ssim = wm_extract(T_model, WMDataset, args, bn_path='./result/bn/AlexNet_C10_FedAvg_wm.h5')
print(f"epoch:{0}, test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f}, SSIM:{avg_ssim:.2f}")
SSIM_list.append(avg_ssim * 100)
SSIM_ATT.append(0)

optimizer = torch.optim.SGD(T_model.parameters(), lr=0.001, momentum=0.9)
epochs = 60
wm, key = split_data.wm, split_data.key_vec
wm, key = wm.unsqueeze(0).to(dtype=torch.float32).to(args.device), key.unsqueeze(0).to(dtype=torch.float32).to(args.device)
for epoch in range(epochs):
    T_model.train()

    for vectors in split_dataloader:
        vectors = vectors.to(dtype=torch.float32).to(args.device)
        rebuild_wms = T_model(vectors)
        loss = compute_loss_wm(
            rebuild_wms=rebuild_wms,
            vectors=vectors,
            key=key,
            wm=wm,
            device=args.device,
            ssim_threshold=0.95,
            ssim_low_target=0.3,
            data_range=1
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_acc, test_ave_loss = evaluate(model, args)
    ACC_list.append(test_acc)

    # avg_ssim = WMextract(T_model, wmDataloader, args, epoch + 1, bn_path = '../result/bn/VGG13_C10_FedAvg_wm.h5')
    build_wms, avg_ssim = wm_extract(T_model, WMDataset, args, bn_path='./result/bn/AlexNet_C10_FedAvg_wm.h5')
    re_wm = T_model(key)
    ssim_a = ssim(re_wm, wm)
    SSIM_ATT.append(ssim_a.item() * 100)
    if epoch == epochs-1:
        # time.sleep(5)
        plt.subplot(1, 2, 1)
        imshow(wm[0])
        plt.title("attacker wm:" + str(epoch + 1))
        plt.subplot(1, 2, 2)
        imshow(re_wm[0])
        plt.title("attacker re_wm:" + str(epoch + 1))
        plt.savefig(f"./result/img/AlexNet_ada_{ssim_a.item():0.2f}.png")
        plt.close()
    print(
        f"epoch:{epoch+1}, test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f}, SSIM:{avg_ssim:.2f}, attacker ssim:{ssim_a:.2f} ")
    SSIM_list.append(avg_ssim * 100)

plt.figure(figsize=(8, 6))
x = np.arange(len(ACC_list))
plt.plot(x, ACC_list, marker='o', linestyle='-', color='b', label='test_acc')
plt.plot(x, SSIM_list, marker='o', linestyle='-', color='g', label='watermark')
plt.plot(x, SSIM_ATT, marker='o', linestyle='-', color='r', label='attacker wm')
plt.title('adaptive attack')  # 设置标题
plt.xlabel('epochs')  # 设置 x 轴标签
plt.ylabel('result')  # 设置 y 轴标签
plt.grid(True)  # 添加网格线
plt.legend()  # 添加图例
plt.savefig(f"./result/img/AlexNet_ada.png")
plt.close()

