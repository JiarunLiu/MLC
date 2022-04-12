import json
import datetime
import numpy as np
from os.path import join

import torch
import torchvision

from dataset.cifar import CIFAR10
from dataset.mnist import MNIST
from dataset.clothing1m import Clothing1M
from models.resnet import resnet18
from models.coteaching_model import MLPNet, CNN


class BasicTrainer(object):

    def __init__(self, args):
        self._get_args(args)
        if self.args.random_seed is not None:
            torch.manual_seed(self.args.random_seed)
            np.random.seed(self.args.random_seed)

    def _save_meta(self):
        print(vars(self.args))
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        with open(join(self.args.dir, "settings-{}.json".format(nowTime)), 'w') as f:
            json.dump(vars(self.args), f, indent=4, sort_keys=True)

    def _get_args(self, args):
        self.args = args
        self.args.checkpoint_dir = join(self.args.dir, "checkpoint.pth.tar")
        self.args.modelbest_dir = join(self.args.dir, "model_best.pth.tar")
        self.args.record_dir = join(self.args.dir, 'record.json')
        self.args.y_file = join(self.args.dir, "y.npy")
        self.best_prec1 = 0

    def _get_model(self, backbone):
        if backbone == 'resnet18':
            model = resnet18(pretrained=True, num_classes=self.args.classnum).to(self.args.device)
        elif backbone == 'mlp':
            model = MLPNet().to(self.args.device)
        elif backbone == "cnn" or backbone == "CNN":
            model = CNN(n_outputs=self.args.classnum, input_channel=self.args.input_dim, linear_num=self.args.linear_num).to(self.args.device)

        return model

    def _get_optim(self, parm, optim="SGD", lr=None):
        if optim.lower() == "sgd":
            optimizer = torch.optim.SGD(parm, lr=lr if lr else self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif optim.lower() == "adam":
            optimizer = torch.optim.Adam(parm, lr=lr if lr else self.args.lr)
        else:
            ValueError("No Such Optimizer Implemented: {}".format(optim))

        return optimizer

    def _get_dataset_clothing1m(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.7213500457070274, 0.6844039100313122, 0.667703515181475),
                                             std=(0.30372145560238617, 0.31379271438793327, 0.3182373132184539)),
        ])

        transform2 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.7263949816798755, 0.6885419850670035, 0.6710302587685353),
                                             std=(0.3026707473687398, 0.3140545091822093, 0.31887224913014806)),
        ])

        transform3 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.7247519698485186, 0.6868922457297535, 0.6692883956444088),
                                             std=(0.7247519698485186, 0.6868922457297535, 0.6692883956444088)),
        ])

        trainset = Clothing1M(root=join(self.args.root, 'noisy_train'),
                              transform=transform,
                              noise_type='clean',
                              noise_rate=0.0,
                              device=self.args.data_device,
                              )
        testset = Clothing1M(root=join(self.args.root, 'clean_test'),
                             transform=transform2,
                             noise_type='clean',
                             noise_rate=0.0,
                             device=self.args.data_device,
                             )
        valset = Clothing1M(root=join(self.args.root, 'clean_val'),
                            transform=transform3,
                            noise_type='clean',
                            noise_rate=0.0,
                            device=self.args.data_device,
                            )

        return trainset, testset, valset

    def _get_dataset_mnist(self):
        transform1 = torchvision.transforms.Compose([
            torchvision.transforms.RandomPerspective(),
            torchvision.transforms.ColorJitter(0.2, 0.75, 0.25, 0.04),
            torchvision.transforms.ToTensor(),
        ])
        transform = torchvision.transforms.ToTensor()
        trainset = MNIST(root=self.args.root,
                         download=True,
                         train=0,
                         transform=transform1,
                         noise_type=self.args.noise_type,
                         noise_rate=self.args.noise,
                         )
        testset = MNIST(root=self.args.root,
                        download=True,
                        train=1,
                        transform=transform,
                        noise_type='clean',
                        noise_rate=0,
                        )
        valset = MNIST(root=self.args.root,
                       download=True,
                       train=2,
                       transform=transform,
                       noise_type='clean',
                       noise_rate=0,
                       )

        return trainset, testset, valset

    def _get_dataset_cifar10(self):
        transform1 = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),
            torchvision.transforms.ToTensor(),
        ])
        transform2 = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        trainset = CIFAR10(root=self.args.root,
                           download=True,
                           train=True,
                           transform=transform1,
                           noise_type=self.args.noise_type,
                           noise_rate=self.args.noise,
                           )
        testset = CIFAR10(root=self.args.root,
                          download=True,
                          train=False,
                          transform=transform2,
                          noise_type='clean',
                          noise_rate=self.args.noise,
                          )
        valset = CIFAR10(root=self.args.root,
                         download=True,
                         train=False,
                         transform=transform2,
                         noise_type='clean',
                         noise_rate=self.args.noise,
                         )
        return trainset, testset, valset

    def _load_data(self):
        if self.args.dataset == 'mnist':
            trainset, testset, valset = self._get_dataset_mnist()
        elif self.args.dataset == 'cifar10':
            trainset, testset, valset = self._get_dataset_cifar10()
        elif self.args.dataset == 'clothing1m':
            trainset, testset, valset = self._get_dataset_clothing1m()
        else:
            ValueError("Wrong Dataset [{}]".format(self.args.dataset))

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True if self.args.data_device == 1 else False
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True if self.args.data_device == 1 else False
        )
        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True if self.args.data_device == 1 else False
        )

        self.train_batch_num = len(trainloader)
        self.test_batch_num = len(testloader)
        self.val_batch_num = len(valloader)

        self.train_data_num = len(trainset)
        self.test_data_num = len(testset)
        self.val_data_num = len(valset)

        self.noise_or_not = trainset.noise_or_not
        self.clean_labels = trainset.labels

        print("Train num: {}\tTest num: {}\tVal num: {}".format(len(trainset), len(testset), len(valset)))
        return trainloader, testloader, valloader