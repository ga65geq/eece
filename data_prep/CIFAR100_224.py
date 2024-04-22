import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

train_size = 40000
val_size = 10000
class_num = 10

def create_dataset(root='../data/CIFAR100', transform_training=transform, transform_test=transform):
    trainset = datasets.CIFAR100(root=root, train=True, download=True, transform=transform_training)
    testset = datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    return trainset, testset

def get_same_category_index(dataset, category):
    if isinstance(dataset, Subset):
        idx = torch.where(torch.tensor(dataset.dataset.targets)[dataset.indices] == category)[0].tolist()
    else:
        idx = torch.where(torch.tensor(dataset.targets) == category)[0].tolist()
    return idx