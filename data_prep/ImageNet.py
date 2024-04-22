import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import os

transform_training = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])

train_size = 1231167
val_size = 50000
class_num = 1000

def create_dataset(root, transform_training=transform_training, transform_test=transform_test):
    train_root = os.path.join(root, 'train')
    test_root = os.path.join(root, 'val')
    trainset = datasets.ImageFolder(train_root, transform_training)
    testset = datasets.ImageFolder(test_root, transform_test)
    return trainset, testset

def get_same_category_index(dataset, category):
    if isinstance(dataset, Subset):
        idx = torch.where(torch.tensor(dataset.dataset.targets)[dataset.indices] == category)[0].tolist()
    else:
        idx = torch.where(torch.tensor(dataset.targets) == category)[0].tolist()
    return idx