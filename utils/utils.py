# -*- coding: UTF-8 -*-

import hashlib
import numpy as np
from torch.utils.data import DataLoader
import h5py
from torch.autograd import Variable
from math import exp
from utils.models import *
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms


def get_optim(model, optim, lr=0.001, momentum=0.9, decay = 0.0001):

    if optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    elif optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    elif optim == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    else:
        exit("Unknown Optimizer!")
        
def get_full_dataset(dataset_name, train=True, img_size=(32, 32), is_download=False):
    if train == True:
        if dataset_name == 'mnist':
            dataset = datasets.MNIST('./data/mnist/', train=True, download=is_download,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Resize(img_size, antialias=False),
                                      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  
                                  ]))
        elif dataset_name == 'fashionmnist':
           
            dataset = datasets.FashionMNIST('./data/fashionmnist/', train=True, download=is_download,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Resize(img_size, antialias=False),
                                                transforms.Lambda(lambda x: x.repeat(3, 1, 1)), 

                                            ]))
        elif dataset_name == 'cifar10':
            dataset = datasets.CIFAR10('./data/cifar10/', train=True, download=is_download,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Pad(4, padding_mode="reflect"),
                                        transforms.RandomCrop(img_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                                    ]))
        elif dataset_name == 'cifar100':
            dataset = datasets.CIFAR100('./data/cifar100/', train=True, download=is_download,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Pad(4, padding_mode="reflect"),
                                        transforms.RandomCrop(img_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                                                             std=(0.2675, 0.2565, 0.2761))
                                    ]))

        else:
            exit("Unknown Dataset")
    else:
        if dataset_name == 'mnist':
            dataset = datasets.MNIST('./data/mnist/', train=False, download=is_download,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Resize(img_size),
                                              transforms.Lambda(lambda x: x.repeat(3, 1, 1)), 
                                          ]))
        elif dataset_name == 'fashionmnist':
           
            dataset = datasets.FashionMNIST('./data/fashionmnist/', train=False, download=is_download,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Resize(img_size, antialias=False),
                                                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  

                                            ]))
        elif dataset_name == 'cifar10':
            dataset = datasets.CIFAR10('./data/cifar10/', train=False, download=is_download,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                                   ]))
        elif dataset_name == 'cifar100':
            dataset = datasets.CIFAR100('./data/cifar100/', train=False, download=is_download,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=(0.5071, 0.4867, 0.4408),
                                                            std=(0.2675, 0.2565, 0.2761))

                                   ]))
        else:
            exit("Unknown Dataset")
    return dataset

def iid_split(dataset, num_clients):
    
    dataset_len = len(dataset)
    num_items = dataset_len // num_clients
    dict_clients = dict()
    all_idxs = [i for i in range(dataset_len)]
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_clients[i])
    return dict_clients

def dniid_split(dataset, num_clients, param=0.8):
    
    dataset_len = len(dataset)
    dataset_y = np.array(dataset.targets)
    labels = set(dataset_y)
    sorted_idxs = dict()
    for label in labels:
        sorted_idxs[label] = []

 
    for i in range(dataset_len):
        sorted_idxs[dataset_y[i]].append(i)

    for label in labels:
        sorted_idxs[label] = np.array(sorted_idxs[label])

   
    dict_clients = dict()
    for i in range(num_clients):
        dict_clients[i] = None

    for label in labels:
        idxs = sorted_idxs[label]
        sample_split = np.random.dirichlet(np.array(num_clients * [param]))
        accum = 0.0
        num_of_current_class = idxs.shape[0]
        for i in range(num_clients):
            client_idxs = idxs[int(accum * num_of_current_class):
                               min(dataset_len, int((accum + sample_split[i]) * num_of_current_class))]
            if dict_clients[i] is None:
                dict_clients[i] = client_idxs
            else:
                dict_clients[i] = np.concatenate((dict_clients[i], client_idxs))
            accum += sample_split[i]
    return dict_clients

def create_T_model(model, model_type):
    if model_type == 'VGG13':
        transposed_model = TransposedVGG13(model)
        return transposed_model
    elif model_type == 'AlexNet':
        transposed_model = TransposedAlexNet(model)
        return transposed_model
    elif model_type == 'ResNet18':
        transposed_model = TransposedResNet18(model)
        return transposed_model
    elif model_type == "MobileNetV2":
        transposed_model = TransposedMobileNetV2(model)
        return transposed_model
    else:
        print("model type not supported!")
def get_model(args):
    if args.model == 'VGG13':
        return VGG13(args.num_classes)
    elif args.model == 'AlexNet':
        return AlexNet(args.num_classes)
    elif args.model == 'ResNet18':
        return ResNet18(args.num_classes)
    elif args.model == "MobileNetV2":
        return MobileNetV2()
    else:
        exit("Unknown Model!")

def evaluate(model, args):
    test_dataset = get_full_dataset(args.dataset, train=False, img_size=(args.image_size, args.image_size))
    test_Loader = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False)
    model.eval()  
    model.to(args.device)
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = []
    with torch.no_grad(): 
        for images, labels in test_Loader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            loss = criterion(outputs, labels)  
            _, predicted = outputs.max(1)  
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()  
            test_loss.append(loss.item())
    accuracy = 100. * correct / total  
    average_loss = np.mean(test_loss)  
    return accuracy, average_loss
    
def wm_extract(T_model, wm_dataset, args, bn_path = './result/forgery/bn/test.h5') :
    wm_loader = DataLoader(wm_dataset, batch_size=1, shuffle=False)
    T_model.eval().to(args.device)
    with torch.no_grad():
        re_wms = []
        ssims = []
        for num, (wm, key, seed) in enumerate(wm_loader):
            key = key.to(dtype=torch.float32).to(args.device)
            wm = wm.to(dtype=torch.float32).to(args.device)
            bn = load_bn_stats_hdf5(seed[0], args.device, bn_path)
            re_wm = T_model(key, bn).to(args.device)
            re_wms.append(re_wm)
            ssim_v = ssim(re_wm, wm)
            ssims.append(ssim_v.item())
        re_wms = torch.cat(re_wms, dim=0)
    return re_wms, np.mean(ssims)

#--------------------SSIM---------------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel,sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def calculate_constants(data_range):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    return C1, C2

def _ssim(img1, img2, window, window_size, channel, size_average=True, data_range=1):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1, C2 = calculate_constants(data_range)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True, data_range=1):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, data_range)

#-----------------------------------------------
def imshow(img):
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.axis('off')
    
def generate_hash_code(seed="A", dim=10):
    bytes_per_dim = 8  
    total_bytes = dim * bytes_per_dim

    shake = hashlib.shake_128()
    shake.update(seed.encode())
    hash_bytes = shake.digest(total_bytes)

    code = np.zeros(dim)
    for i in range(dim):
        start = i * bytes_per_dim
        end = start + bytes_per_dim
        segment = hash_bytes[start:end]
        int_val = int.from_bytes(segment, byteorder='big')
        max_int = 2 ** (8 * bytes_per_dim) - 1
        code[i] = 2 * (int_val / max_int) - 1 
    return code
    

def save_bn_stats_hdf5(clients, save_path="./bn_stats.h5"):
    with h5py.File(save_path, 'w') as f:
        for client_id, client in enumerate(clients):
            client_group = f.create_group(f"client_{client.seed}")
            for name, stats in client.bn_stats.items():
                layer_group = client_group.create_group(name)
                layer_group.create_dataset('running_mean', data=stats['running_mean'].clone().cpu().numpy())
                layer_group.create_dataset('running_var', data=stats['running_var'].clone().cpu().numpy())
    print(f"All BN stats saved to {save_path}")
    
def load_bn_stats_hdf5(client_label, device, save_path="./bn_stats.h5"):
    bn_stats = {}
    with h5py.File(save_path, 'r') as f:
        client_group = f.get(f"client_{client_label}")
        if client_group is None:
            print(f"No BN stats found for client {client_label}")
            return None
        for name in client_group.keys():
            layer_group = client_group[name]
            bn_stats[name] = {
                'running_mean': torch.tensor(layer_group['running_mean'][:]).to(device),
                'running_var': torch.tensor(layer_group['running_var'][:]).to(device)
            }
    return bn_stats




