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
                                      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 单通道转三通道
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
                                              transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 单通道转三通道
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
    """
    Split I.I.D client data
    :param dataset:
    :param num_clients:
    :return: dict of image indexes，
        dict_clients：一个字典，其键是客户编号（从0到num_clients-1），值是每个客户获得的数据索引集合。
    """

    dataset_len = len(dataset)
    num_items = dataset_len // num_clients
    dict_clients = dict()
    all_idxs = [i for i in range(dataset_len)]
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_clients[i])
    return dict_clients

def dniid_split(dataset, num_clients, param=0.8):
    """
    Using Dirichlet distribution to sample non I.I.D client data
    :param dataset:
    :param num_clients:
    :param param: parameter used in Dirichlet distribution
    :return: dict of image indexes
    """
    dataset_len = len(dataset)
    dataset_y = np.array(dataset.targets)
    labels = set(dataset_y)
    sorted_idxs = dict()
    for label in labels:
        sorted_idxs[label] = []

    # sort indexes by labels
    for i in range(dataset_len):
        sorted_idxs[dataset_y[i]].append(i)

    for label in labels:
        sorted_idxs[label] = np.array(sorted_idxs[label])

    # initialize the clients' dataset dict
    dict_clients = dict()
    for i in range(num_clients):
        dict_clients[i] = None
    # split the dataset separately
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

def pniid_split(dataset, num_clients, num_of_shards_each_clients=2):
    """
    Simulate pathological non I.I.D distribution
    :param dataset:
    :param num_clients:
    :param num_of_shards_each_clients:
    :return:
    """
    dataset_len = len(dataset)
    dataset_y = np.array(dataset.targets)

    sorted_idxs = np.argsort(dataset_y)

    size_of_each_shards = dataset_len // (num_clients * num_of_shards_each_clients)
    per = np.random.permutation(num_clients * num_of_shards_each_clients)
    dict_clients = dict()
    for i in range(num_clients):
        idxs = np.array([])
        for j in range(num_of_shards_each_clients):
            idxs = np.concatenate((idxs, sorted_idxs[per[num_of_shards_each_clients * i + j] * size_of_each_shards:
                                   min(dataset_len, (per[num_of_shards_each_clients * i + j] + 1) * size_of_each_shards)]))
        dict_clients[i] = idxs
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
    else:
        print("model type not supported!")
def get_model(args):
    if args.model == 'VGG13':
        return VGG13(args.num_classes)
    elif args.model == 'AlexNet':
        return AlexNet(args.num_classes)
    elif args.model == 'ResNet18':
        return ResNet18(args.num_classes)
    else:
        exit("Unknown Model!")

def evaluate(model, args):
    test_dataset = get_full_dataset(args.dataset, train=False, img_size=(32, 32))
    test_Loader = DataLoader(test_dataset, batch_size=args.test_bs, shuffle=False)
    model.eval()  # 将模型设置为评估模式
    model.to(args.device)
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = []
    with torch.no_grad():  # 在测试过程中禁用梯度计算
        for images, labels in test_Loader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = model(images)
            loss = criterion(outputs, labels)  # 计算损失
            _, predicted = outputs.max(1)  # 获取预测的类别索引
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()  # 计算预测正确的样本数
            test_loss.append(loss.item())
    accuracy = 100. * correct / total  # 计算准确率
    average_loss = np.mean(test_loss)  # 计算平均损失
    return accuracy, average_loss
def wm_extract(T_model, wm_dataset, args, bn_path = '../result/bn/VGG13bn_stats.h5') :
    wm_loader = DataLoader(wm_dataset, batch_size=1, shuffle=False)
    T_model.eval()
    with torch.no_grad():
        re_wms = []
        ssims = []
        for num, (wm, key, seed) in enumerate(wm_loader):
            key = key.to(dtype=torch.float32).to(args.device)
            wm = wm.to(dtype=torch.float32).to(args.device)
            bn = load_bn_stats_hdf5(seed[0], args.device, bn_path)
            re_wm = T_model(key, bn)
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
    # img = 0.5 * img + 0.5  # 反归一化
    plt.imshow(img)
    plt.axis('off')
def generate_hash_code(seed="A", dim=10):
    bytes_per_dim = 8  # 每维使用 8 字节（64位整数）
    total_bytes = dim * bytes_per_dim

    # 使用 shake_128 输出足够的哈希字节
    shake = hashlib.shake_128()
    shake.update(seed.encode())
    hash_bytes = shake.digest(total_bytes)

    # 构造向量
    code = np.zeros(dim)
    for i in range(dim):
        start = i * bytes_per_dim
        end = start + bytes_per_dim
        segment = hash_bytes[start:end]
        int_val = int.from_bytes(segment, byteorder='big')
        max_int = 2 ** (8 * bytes_per_dim) - 1
        code[i] = 2 * (int_val / max_int) - 1  # 映射到 [-1, 1]
    return code
def cosine_similarity(x1, base):
    x1_norm = torch.norm(x1, p=2, dim=1, keepdim=True) + 1e-8
    base_norm = torch.norm(base, p=2, dim=1, keepdim=True) + 1e-8
    dot_product = torch.sum(x1 * base, dim=1, keepdim=True)
    return dot_product / (x1_norm * base_norm)  # [batch, 1]

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

def RandVec_generate(dim, num_vectors, device):
    # Define the number of distributions
    num_distributions = 6  # normal, uniform, bernoulli, poisson, exponential, laplace
    # Calculate vectors per distribution
    vectors_per_dist = num_vectors // num_distributions
    remainder = num_vectors % num_distributions

    vectors = []
    # Normal distribution
    count = vectors_per_dist + (1 if remainder > 0 else 0)
    if count > 0:
        vectors.append(torch.randn(count, dim).to(device))
        remainder -= 1

    # Uniform distribution
    count = vectors_per_dist + (1 if remainder > 0 else 0)
    if count > 0:
        vectors.append(torch.rand(count, dim).to(device))
        remainder -= 1

    # Bernoulli distribution
    count = vectors_per_dist + (1 if remainder > 0 else 0)
    if count > 0:
        vectors.append(torch.bernoulli(torch.full((count, dim), 0.5)).to(device))
        remainder -= 1

    # Poisson distribution
    count = vectors_per_dist + (1 if remainder > 0 else 0)
    if count > 0:
        vectors.append(torch.poisson(torch.full((count, dim), 2.0)).to(device))
        remainder -= 1

    # Exponential distribution
    count = vectors_per_dist + (1 if remainder > 0 else 0)
    if count > 0:
        vectors.append(torch.empty(count, dim).exponential_(lambd=1).to(device))
        remainder -= 1

    # Laplace distribution
    count = vectors_per_dist + (1 if remainder > 0 else 0)
    if count > 0:
        laplace_dist = torch.distributions.Laplace(loc=0, scale=1)
        vectors.append(laplace_dist.sample((count, dim)).to(device))

    # Concatenate all vectors along the batch dimension
    vector_rand = torch.cat(vectors, dim=0)
    return vector_rand[:num_vectors]
#---------------------------------------------------------
def collect_bn_stats(model):
    bn_stats = {'running_mean': [], 'running_var': []}
    for name, module in model.model1.named_modules():
        if isinstance(module, TransposedBatchNorm):
            bn_stats['running_mean'].append(module.running_mean.clone().cpu().numpy())
            bn_stats['running_var'].append(module.running_var.clone().cpu().numpy())
    return bn_stats
def topk_pruning(model, k_percent, model1 = None):
    """
    对模型的所有参数进行 top-k% 剪枝，基于参数绝对值大小。
    Args:
        model: PyTorch 模型 (nn.Module)
        k_percent: 剪枝百分比（0到100之间的浮点数）
    Returns:
        pruned_model: 剪枝后的模型
    """
    # 收集所有参数的绝对值和对应张量
    param_abs = []
    param_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_abs.append(param.abs().detach().view(-1))
            param_names.append(name)

    # 合并所有参数到一个向量
    all_params = torch.cat(param_abs)

    # 计算 top-k% 阈值
    k = int(len(all_params) * (k_percent / 100.0))
    threshold = torch.topk(all_params, k, largest=False)[0][-1]

    # 创建剪枝后的模型副本
    # if model1 is not None:
    #     pruned_model = type(model)(model1).to(next(model.parameters()).device)
    # else:
    #     pruned_model = type(model)().to(next(model.parameters()).device)
    pruned_state_dict = OrderedDict()
    # 获取原始模型的完整状态字典（包括参数和缓冲区）
    for name, param in model.state_dict().items():
        pruned_state_dict[name] = param.clone().detach()  # 深拷贝张量
    # 对每个参数应用剪枝
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 创建掩码：保留绝对值大于阈值的参数
            mask = param.abs().detach() > threshold
            pruned_param = param.clone().detach() * mask.float()
            pruned_state_dict[name] = pruned_param
        else:
            pruned_state_dict[name] = param

    # 加载剪枝后的参数
    # pruned_model.load_state_dict(pruned_state_dict)

    return pruned_state_dict
def check_parameter_sharing(model1, model2):
    """
    验证两个模型的参数是否共享（内存地址相同）。

    Args:
        model1: 第一个 PyTorch 模型 (nn.Module)
        model2: 第二个 PyTorch 模型 (nn.Module)
    """
    # print("Checking parameter sharing:")
    shared_count = 0
    total_count = 0

    # 获取两个模型的参数并转换为字典
    params1 = {name: param for name, param in model1.named_parameters()}
    params2 = {name: param for name, param in model2.named_parameters()}

    # 确保参数名称相同
    if set(params1.keys()) != set(params2.keys()):
        print(f"Warning: Models have different parameter names")
        return

    # 逐一比较参数
    for name in params1:
        param1 = params1[name]
        param2 = params2[name]
        total_count += 1
        if param1 is param2:
            # print(f"Shared parameter: {name} (same memory address)")
            shared_count += 1
        # else:
        #     print(f"Not shared: {name} (different memory address)")

    print(f"Summary: {shared_count}/{total_count} parameters are shared")
