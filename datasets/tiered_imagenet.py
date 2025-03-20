import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .datasets import register


@register('tiered-imagenet')
class TieredImageNet(Dataset):
    def __init__(self, root_path, split='train', image_size=80, **kwargs):
        self.data = []
        self.label = []
        self.class_to_idx = {}

        # 构建类别到索引的映射，并收集数据路径与标签
        for class_id, class_name in enumerate(sorted(os.listdir(os.path.join(root_path, split)))):
            class_path = os.path.join(root_path, split, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = class_id
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.data.append(img_path)
                self.label.append(class_id)

        self.n_classes = len(self.class_to_idx)

        # 保持原始transform逻辑不变
        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        augment = kwargs.get('augment')
        if augment == 'resize':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment is None:
            self.transform = self.default_transform

        #
        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean

        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.label[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label