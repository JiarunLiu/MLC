import os
import os.path
from os.path import join

import numpy as np
from PIL import Image
import torch.utils.data
import torchvision.transforms as transforms

from dataset.utils import noisify


class Clothing1M(torch.utils.data.Dataset):

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 noise_type='clean',
                 noise_rate=0.00,
                 device=0,
                 ):
        self.root = root
        self.class_num = 14
        self.transform = transform
        self.target_transform = target_transform
        self.device = device  # 0: hardware; 1: RAM
        self.noise_type = noise_type
        self.random_state = 0

        img_folder_list = os.listdir(root)
        self.data = []
        self.labels = []
        for label_ in img_folder_list:
            imgs = os.listdir(join(root, label_))
            for img in imgs:
                if img[-3:] == 'jpg':
                    data_ = join(root, label_, img)
                    assert os.path.isfile(data_)
                    if self.device == 1:
                        data_ = self.img_loader(data_)
                    self.data.append(data_)
                    self.labels.append(int(label_))
        self.labels = np.asarray(self.labels)

        if self.device == 1:
            self.data = np.concatenate(self.data)
        else:
            self.data = np.array(self.data)

        if self.device != 1:
            self.data = self.data.tolist()

        # noisy labels
        if noise_type == 'clean':
            self.noise_or_not = np.ones([len(self.labels)], dtype=np.bool)
        else:
            self.noisy_labels, self.actual_noise_rate = noisify(dataset="clothing1m",
                                                                nb_classes=14,
                                                                train_labels=np.expand_dims(self.labels, 1),
                                                                noise_type=noise_type,
                                                                noise_rate=noise_rate,
                                                                random_state=self.random_state)
            self.noisy_labels = self.noisy_labels.squeeze()
            self.noise_or_not = self.noisy_labels == self.labels

    def img_loader(self, img_path):
        return np.asarray(Image.open(img_path).convert("RGB")).astype(np.uint8)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target，index) where target is index of the target class.
        """
        img = self.img_loader(self.data[index])
        target = self.labels[index] if self.noise_type == 'clean' else self.noisy_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.labels)