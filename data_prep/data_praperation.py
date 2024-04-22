import os
import pickle

import torch
from torch.utils.data import DataLoader, Subset, random_split


# Function to save data to a file using pickle
def save_data(filename, data):
    with open(filename, "wb") as file:
        pickle.dump(data, file)
    print(f"Data saved to {filename}")


# Function to load data from a file using pickle
def load_data(filename):
    with open(filename, "rb") as file:
        loaded_data = pickle.load(file)
    print(f"Data loaded from {filename}")
    return loaded_data


class DataPrep:
    def __init__(self, root, name):
        if name == "CIFAR10":
            import data_prep.CIFAR10 as DATA
        elif name == "CIFAR10_224":
            import data_prep.CIFAR10_224 as DATA
        elif name == "CIFAR100":
            import data_prep.CIFAR100 as DATA
        elif name == "CIFAR100_224":
            import data_prep.CIFAR100_224 as DATA
        elif name == "ImageNet":
            import data_prep.ImageNet as DATA
        else:
            raise NotImplementedError("This dataset is not supported!")
        train_set, self.testset = DATA.create_dataset(root)

        split_path_train = "data_split/" + name + "_train.pkl"
        split_path_val = "data_split/" + name + "_val.pkl"
        if os.path.exists("./" + split_path_train) and os.path.exists(
            "./" + split_path_val
        ):
            train_indices, val_indices = (
                load_data("./" + split_path_train),
                load_data("./" + split_path_val),
            )
            self.trainset = Subset(train_set, train_indices)
            self.valset = Subset(train_set, val_indices)
        else:
            self.trainset, self.valset = random_split(
                train_set,
                [
                    DATA.train_size / (DATA.train_size + DATA.val_size),
                    DATA.val_size / (DATA.train_size + DATA.val_size),
                ],
            )
            save_data("./" + split_path_train, self.trainset.indices)
            save_data("./" + split_path_val, self.valset.indices)

        self.get_same_category_index = DATA.get_same_category_index
        self.class_num = DATA.class_num
        print("Create DataPrep for {}".format(name))

    @staticmethod
    def dataset_split(train_size, val_size):
        index = torch.randperm(train_size + val_size)
        index_training = index[:train_size]
        index_val = index[train_size:]
        return index_training, index_val

    def get_dataloader(self, batch_size=128, num_workers=0):
        train_loader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.valset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            self.testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        print("Successfully get dataloader.")
        return train_loader, val_loader, test_loader

    def get_category_index(self, category):
        train_idx = self.get_same_category_index(self.trainset, category)
        val_idx = self.get_same_category_index(self.valset, category)
        test_idx = self.get_same_category_index(self.testset, category)
        return train_idx, val_idx, test_idx

    def get_sub_loader(
        self, train_idx, val_idx, test_idx, num_workers=1, batch_size=100
    ):
        train_loader = DataLoader(
            Subset(self.trainset, train_idx),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        val_loader = DataLoader(
            Subset(self.valset, val_idx),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        test_loader = DataLoader(
            Subset(self.testset, test_idx),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        return train_loader, val_loader, test_loader
