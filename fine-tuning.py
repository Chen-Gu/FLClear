import torch.optim

from utils.parameters import load_args
from utils.utils import *
from utils.datasets import WMDataset

args = load_args()
train_dataset = get_full_dataset(args.dataset, train=True, img_size=(32, 32))
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

wmDataset = WMDataset(image_dir='./data/logo10/', dim = args.num_classes, transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ]))

init_state = torch.load('./result/model/AlexNet_C10_FedAvg_wm.pth', weights_only=False)
model = AlexNet().to(args.device)
model.load_state_dict(init_state)
T_model =  TransposedAlexNet(model).to(args.device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8)

ACC_list = []
SSIM_list = []
test_acc, test_ave_loss = evaluate(model, args)
ACC_list.append(test_acc)

build_wms, avg_ssim = wm_extract(T_model, wmDataset, args, bn_path = './result/bn/AlexNet_C10_FedAvg_wm.h5')
print(f"epoch:{0}, test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f},SSIM:{avg_ssim:.2f}")

SSIM_list.append(avg_ssim * 100)
epochs_fine = 30
for epoch in range(epochs_fine):
    model.train()
    T_model.train()
    criterion = nn.CrossEntropyLoss()

    for images, labels in train_dataloader:
        images, labels = images.to(args.device), labels.to(args.device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_acc, test_ave_loss = evaluate(model, args)
    ACC_list.append(test_acc)
    build_wms, avg_ssim = wm_extract(T_model, wmDataset, args, bn_path = './result/bn/AlexNet_C10_FedAvg_wm.h5')
    print(f"epoch:{epoch+1}, test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f},SSIM:{avg_ssim:.2f}")
    SSIM_list.append(avg_ssim * 100)

    # plt.figure(figsize=(6, 1))
    # for j in range(build_wms.shape[0]):
    #     plt.subplot(1, build_wms.shape[0], j + 1)
    #     imshow(build_wms[j])
    #     if j == 0:
    #         plt.title("fine-tuning epoch:" + str(epoch+1))
    # plt.show()

# plt.figure(figsize=(8, 6))
# x = np.arange(len(ACC_list))
# plt.plot(x, ACC_list, marker='o', linestyle='-', color='b', label='test_acc')
# plt.plot(x, SSIM_list, marker='o', linestyle='-', color='g', label='watermark')
# plt.title('fine-tuning')  # 设置标题
# plt.xlabel('fine-tuning epochs')  # 设置 x 轴标签
# plt.ylabel('result')  # 设置 y 轴标签
# plt.grid(True)  # 添加网格线
# plt.legend()  # 添加图例
# plt.show()