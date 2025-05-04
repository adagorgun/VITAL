from __future__ import print_function

import os
import os.path
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Set the appropriate paths of the datasets here.
_IMAGENET_DATASET_DIR = " " # Imagenet directory

_MEAN_PIXEL = [0.485, 0.456, 0.406]
_STD_PIXEL = [0.229, 0.224, 0.225]

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds

class ImageNetBase(data.Dataset):
    def __init__(
        self,
        data_dir=_IMAGENET_DATASET_DIR,
        split='train',
        split_N=None,
        transform=None):
        assert (split in ('train', 'val')) or (split.find('train_subset') != -1) or (split.find('train_class_subset') != -1) or (split.find('val_class_subset') != -1)
        self.split = split
        self.name = f'ImageNet_Split_' + self.split

        print(f'==> Loading ImageNet dataset - split {self.split}')
        print(f'==> ImageNet directory: {data_dir}')

        self.transform = transform
        print(f'==> transform: {self.transform}')
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        split_dir = train_dir if (self.split.find('train') != -1) else val_dir

        self.data = datasets.ImageFolder(split_dir, self.transform)
        self.labels = [item[1] for item in self.data.imgs]

        # Define the train subset (take only the first split_N samples from each class)
        if self.split.find('train_subset') != -1:
            subsetK = split_N
            assert subsetK > 0
            self.split = 'train'

            label2ind = buildLabelIndex(self.data.targets)
            all_indices = []
            for label, img_indices in label2ind.items():
                assert len(img_indices) >= subsetK
                all_indices += img_indices[:subsetK]

            self.data.imgs = [self.data.imgs[idx] for idx in  all_indices]
            self.data.samples = [self.data.samples[idx] for idx in  all_indices]
            self.data.targets = [self.data.targets[idx] for idx in  all_indices]
            self.labels = [self.labels[idx] for idx in  all_indices]

        # Define the train subset (take the samples only from the defined classes)
        if self.split.find('train_class_subset') != -1:
            self.split = 'train'
            class_id = split_N
            label2ind = buildLabelIndex(self.data.targets)
            all_indices = []
            for label, img_indices in label2ind.items():
                if label in class_id:
                    all_indices += img_indices

            self.data.imgs = [self.data.imgs[idx] for idx in  all_indices]
            self.data.samples = [self.data.samples[idx] for idx in  all_indices]
            self.data.targets = [self.data.targets[idx] for idx in  all_indices]
            self.labels = [self.labels[idx] for idx in  all_indices]

        # Define the validation subset (take the samples only from the defined classes)
        if self.split.find('val_class_subset') != -1:
            self.split = 'val'
            class_id = split_N
            label2ind = buildLabelIndex(self.data.targets)
            all_indices = []
            for label, img_indices in label2ind.items():
                if label in class_id:
                    all_indices += img_indices

            self.data.imgs = [self.data.imgs[idx] for idx in  all_indices]
            self.data.samples = [self.data.samples[idx] for idx in  all_indices]
            self.data.targets = [self.data.targets[idx] for idx in  all_indices]
            self.labels = [self.labels[idx] for idx in  all_indices]

    def __getitem__(self, index):
        img, label = self.data[index]
        return img, label

    def __len__(self):
        return len(self.data)


class ImageNet(ImageNetBase):
    def __init__(
        self,
        data_dir=_IMAGENET_DATASET_DIR,
        split='train',
        split_N=None):

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN_PIXEL, std=_STD_PIXEL),
        ])
        ImageNetBase.__init__(self, data_dir=data_dir, split=split, split_N=split_N, transform=transform)

if __name__ == '__main__':
    dataset = ImageNet(split='train_class_subset', split_N=[0, 10])
