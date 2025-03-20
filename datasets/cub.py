import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .datasets import register


@register('cub')
class CUB(Dataset):
    def __init__(self, root_path, split='train', image_size=80, **kwargs):
        # print(f'root_path={root_path}')
        # print(os.listdir(root_path))
        # print(os.listdir(os.path.join(root_path, 'images')))

        isTrain = []
        with open(os.path.join(root_path, 'train_test_split.txt'), 'r', encoding='utf-8') as file:
            for line in file:
                isTrain.append(line.strip().split(' ')[1])

        # print('len=', len(isTrain))

        self.data = []
        self.label = []
        self.class_to_idx = {}

        i = 0
        # 构建类别到索引的映射，并收集数据路径与标签
        for class_id, class_name in enumerate(sorted(os.listdir(os.path.join(root_path, 'images')))):

            class_path = os.path.join(root_path, 'images', class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = class_id
            for img_name in os.listdir(class_path):
                if (split == 'train' and isTrain[i] == '1') or (split == 'test' and isTrain[i] == '0'):
                    img_path = os.path.join(class_path, img_name)
                    self.data.append(img_path)
                    self.label.append(class_id)
                i += 1

        # print(f'i={i}')

        self.n_classes = len(set(self.label))

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

    def find_classes(self):
        classes = []
        class_to_idx = {}
        with open(self.annotations_file, 'r') as f:
            for line in f:
                class_id, class_name = line.split(' ')
                classes.append(class_name)
                class_to_idx[class_name] = int(class_id) - 1  # Adjust for zero-indexing
        return classes, class_to_idx

    def make_dataset(self):
        images = []
        labels = []
        with open(self.split_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                img_id, img_name, is_train = row
                if is_train == '1' and self.split == 'train' or is_train == '0' and self.split == 'test':
                    img_path = os.path.join(self.image_folder, img_name)
                    label = self.class_to_idx[img_name.split('/')[0]]  # Assuming folder name corresponds to class
                    images.append(img_path)
                    labels.append(label)
        return images, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.label[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, label
