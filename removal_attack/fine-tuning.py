import torch.optim

from utils.parameters import load_args
from utils.utils import *
from utils.datasets import WMDataset

args = load_args()
train_dataset = get_full_dataset(args.dataset, train=True, img_size=(32, 32))
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

wmDataset = WMDataset(image_dir='../data/logo10/', dim = args.num_classes, transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                ]))

save_file = 'test' 
init_state = torch.load('../result/forgery/model/'+ save_file +'.pth', weights_only=False)
model = MobileNetV2().to(args.device)
model.load_state_dict(init_state)
T_model =  TransposedMobileNetV2(model).to(args.device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
    build_wms, avg_ssim = wm_extract(T_model, wmDataset, args, bn_path = '../result/'+ save_file +'.h5')
    print(f"epoch:{epoch+1}, test accuracy: {test_acc:.2f}%, average loss: {test_ave_loss:.2f},SSIM:{avg_ssim:.2f}")

