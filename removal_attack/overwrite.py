import time

import matplotlib.pyplot as plt
import torch.optim

from fed.client import compute_loss_wm
from utils.parameters import load_args
from utils.utils import *
from utils.datasets import WMDataset, WMDatasetSplitAndExpend


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

attackedDataset = WMDataset(image_dir='../data/other/animal/', transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
split_data = WMDatasetSplitAndExpend(attackedDataset, [1,])
split_dataloader = DataLoader(split_data, batch_size=32, shuffle=False)
attackedDataloader = DataLoader(attackedDataset, batch_size=1, shuffle=False)

WMDataset = WMDataset(image_dir='../data/attacker/', transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
wmDataloader = DataLoader(WMDataset, batch_size=1, shuffle=False)

save_file = 'test' 
init_state = torch.load('../result/'+ save_file +'.pth', weights_only=False)
model = ResNet18()
model.load_state_dict(init_state)
T_model =  TransposedResNet18(model)

optimizer = torch.optim.SGD(T_model.parameters(), lr=0.001, momentum=0.9)
epochs = 60
wm, key = split_data.wm, split_data.vector
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
    build_wms, avg_ssim = wm_extract(T_model, wmDataset, args, bn_path='../result/'+ save_file +'.h5')
    
    T_model.eval()
    re_wm = T_model(key)
    ssim_a = ssim(re_wm, wm)
    print(f"epoch:{epoch+1}, test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f}, SSIM:{avg_ssim:.2f}, attacker ssim:{ssim_a:.2f} ")
  



