import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

transform_training = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                              (0.247, 0.243, 0.261))])

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                          (0.247, 0.243, 0.261))])
train_size = 40000
val_size = 10000
class_num = 10


def create_dataset(root='../data/CIFAR10', transform_training=transform_training, transform_test=transform_test):
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_training)
    testset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return trainset, testset


def get_same_category_index(dataset, category):
    if isinstance(dataset, Subset):
        idx = torch.where(torch.tensor(dataset.dataset.targets)[dataset.indices] == category)[0].tolist()
    else:
        idx = torch.where(dataset.targets == category)[0]
    return idx
