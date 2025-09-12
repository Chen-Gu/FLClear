import numpy as np
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

from utils.utils import generate_hash_code, cosine_similarity
from torchvision.datasets import ImageFolder

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
# class WMDatasetSplit(Dataset):
#     def __init__(self, wmdataset, widxs):
#         self.wm_dataset = wmdataset
#         self.w_idxs = list(widxs)
#
#     def __len__(self):
#         return len(self.w_idxs)
#
#     def __getitem__(self, item):
#         return self.wm_dataset[self.w_idxs[item]]
class VectorAugment(Dataset):
    def __init__(self, wm_dataset, wm_idx, dim):
        if len(wm_idx) != 1:
            raise ValueError("wm_idx must contain exactly one index")
        self.wm_dataset = wm_dataset
        self.dim = dim
        self.w_idx = list(wm_idx)[0]  # 提取单个索引
        self.wm, self.key_vec, self.seed = self.wm_dataset[self.w_idx]

        if self.wm is None or self.key_vec is None:
            raise ValueError("Invalid wm or key from wm_dataset")

        self.sigma_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.1,0.3,0.5,0.7,0.9,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
        self.vectors = self._generate_samples()

    def _generate_samples(self):

        noise = []
        for sigma in self.sigma_list:
            temp = torch.normal(mean=0, std=sigma, size=(10, self.dim))
            noise.append(temp)
        noise = torch.cat(noise, dim=0)  # [64, 10]
        vector_sets = torch.cat((self.key_vec + noise, self.key_vec.unsqueeze(0)), dim=0).to(dtype=torch.float32)  # [65, 10]

        return vector_sets

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, item):
        return self.vectors[item]

# class WMDataset(Dataset):
#     def __init__(self, image_dir, transform = None):
#         """
#         初始化数据集
#         :param image_dir: 包含图片的目录路径
#         :param transform: 用于图片转换的可选操作
#         """
#         self.image_dir = image_dir
#         self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#
#         image_path = self.image_paths[idx]
#         if self.image_dir == '../data/Watermark_data1/':
#             if '1.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a black A", dim=10)
#                 label = 'A'
#             elif '2.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a black B", dim=10)
#                 label = 'B'
#             elif '3.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a black C", dim=10)
#                 label = 'C'
#             elif '4.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a black D", dim=10)
#                 label = 'D'
#             elif '5.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a blue W", dim=10)
#                 label = 'W'
#             elif '6.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a black F", dim=10)
#                 label = 'F'
#             elif '7.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a red S", dim=10)
#                 label = 'S'
#             elif '8.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a black H", dim=10)
#                 label = 'H'
#             elif '9.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a blue Y", dim=10)
#                 label = 'Y'
#             else:
#                 key = generate_hash_code(seed="this is a black K", dim=10)
#                 label = 'K'
#         else:
#             if '1.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a black Awdew", dim=10)
#                 label = 'A'
#             elif '2.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a black Bcarfwf", dim=10)
#                 label = 'B'
#             elif '3.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a black Chyttj", dim=10)
#                 label = 'C'
#             elif '4.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a black Durrgrt", dim=10)
#                 label = 'D'
#             elif '5.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a blue Wyhtetr", dim=10)
#                 label = 'W'
#             elif '6.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a black Ffdhdr", dim=10)
#                 label = 'F'
#             elif '7.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a red Seergcb", dim=10)
#                 label = 'S'
#             elif '8.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a black Hqaebjy", dim=10)
#                 label = 'H'
#             elif '9.jpg' in image_path:
#                 key = generate_hash_code(seed="this is a blue Yadxzhyr", dim=10)
#                 label = 'Y'
#             else:
#                 key = generate_hash_code(seed="this is a black Khrtzda", dim=10)
#                 label = 'K'
#         key = torch.tensor(key)
#         image = Image.open(image_path).convert('RGB')  # 确保图片是RGB格式
#         image = self.transform(image)
#         return image, key, label

class WMDataset(Dataset):
    def __init__(self, image_dir, dim, transform=None):
        """
        初始化数据集，使用 ImageFolder 加载图片。

        :param root_dir: 包含以标签命名的子文件夹的总文件夹路径
        :param transform: 用于图片转换的可选操作
        """
        self.root_dir = image_dir
        self.transform = transform
        self.dim = dim
        # 使用 ImageFolder 加载数据集
        self.image_folder = ImageFolder(image_dir, transform=None)  # 延迟应用 transform
        self.seeds = self.image_folder.classes  # 获取标签列表（如 ['A', 'B', 'C', ...]）

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        # 获取图片和标签
        image, _ = self.image_folder[idx]  # ImageFolder 返回 (image, class_idx)
        seed = self.seeds[self.image_folder.targets[idx]]  # 获取字符串标签

        # 生成密钥向量
        key = generate_hash_code(seed=seed, dim=self.dim)  # 使用标签作为种子
        key = torch.tensor(key, dtype=torch.float32)  # 转换为浮点型张量

        if self.transform:
            image = self.transform(image)

        return image, key, seed


#--------- waffle 方法 --------------
class WafflePatternDataset(Dataset):
    def __init__(self, args, path='../data/pattern'):

        self.num_classes = args.num_classes
        self.image_size = args.image_size
        self.num_channels = args.num_channels
        self.num_trigger_set = args.num_trigger_set
        self.path = path

        self.trigger_imgs, self.labels = self.generate_waffle_pattern()

    def __len__(self):
        return len(self.trigger_imgs)

    def generate_waffle_pattern(self):

        base_patterns = []
        for i in range(self.num_classes):
            pattern_path = os.path.join(self.path, "{}.png".format(i))
            if not os.path.exists(pattern_path):
                raise FileNotFoundError(f"Pattern file {pattern_path} not found")
            pattern = Image.open(pattern_path)
            if self.num_channels == 1:
                pattern = pattern.convert("L")
            else:
                pattern = pattern.convert("RGB")
            pattern = np.array(pattern)
            pattern = np.resize(pattern, (self.image_size, self.image_size, self.num_channels))
            base_patterns.append(pattern)

        # Generate trigger set
        trigger_set = []
        trigger_set_labels = []
        num_trigger_each_class = self.num_trigger_set // self.num_classes
        for label, pattern in enumerate(base_patterns):
            for _ in range(num_trigger_each_class):
                image = (pattern + np.random.randint(0, 255, (self.image_size, self.image_size, self.num_channels))) \
                            .astype(np.float32) / 255 / 2
                trigger_set.append(image)
                trigger_set_labels.append(label)

        trigger_set = np.array(trigger_set)
        trigger_set_labels = np.array(trigger_set_labels)

        # Calculate mean and std for normalization
        trigger_set_mean = np.mean(trigger_set, axis=(0, 1, 2))
        trigger_set_std = np.std(trigger_set, axis=(0, 1, 2))
        print(f"Trigger set mean: {trigger_set_mean}, std: {trigger_set_std}")

        # Apply transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(trigger_set_mean, trigger_set_std)
        ])
        images = []
        for img in trigger_set:
            img_tensor = transform(img)
            images.append(img_tensor)
        images = torch.stack(images)
        labels = torch.tensor(trigger_set_labels, dtype=torch.long)
        return images, labels

    def __getitem__(self, item):
        image = self.trigger_imgs[item]
        label = self.labels[item]
        return image, label


