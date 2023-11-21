import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Copied and adjusted from
# https://github.com/yobibyte/unitary-scalarization-dmtl/blob/main/supervised_experiments/loaders/multi_mnist_loader.py
class MultiMNIST(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    multi_training_file = 'multi_training.pt'
    multi_validation_file = 'multi_validation.pt'
    multi_test_file = 'multi_test.pt'

    def __init__(self, root, split="train", transform=None):
        assert split in ["train", "val", "test"]

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split

        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not self._check_multi_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download MNIST and generate a random MultiMNIST')
        if self.split == "train":
            self.train_data, self.train_labels_l, self.train_labels_r = torch.load(
                os.path.join(self.root, self.processed_folder, self.multi_training_file),
                map_location=map_location)
        elif self.split == "val":
            self.validation_data, self.validation_labels_l, self.validation_labels_r = torch.load(
                os.path.join(self.root, self.processed_folder, self.multi_validation_file), 
                map_location=map_location)
        else:
            self.test_data, self.test_labels_l, self.test_labels_r = torch.load(
                os.path.join(self.root, self.processed_folder, self.multi_test_file), 
                map_location=map_location)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == "train":
            img, target_l, target_r = self.train_data[index], self.train_labels_l[index], self.train_labels_r[index]
        elif self.split == "val":
            img, target_l, target_r = self.validation_data[index], self.validation_labels_l[index], \
                                      self.validation_labels_r[index]
        else:
            img, target_l, target_r = self.test_data[index], self.test_labels_l[index], self.test_labels_r[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.cpu().numpy().astype(np.uint8), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        return img, {'L': target_l, 'R': target_r}

    def __len__(self):
        if self.split == "train":
            return len(self.train_data)
        elif self.split == "val":
            return len(self.validation_data)
        else:
            return len(self.test_data)

    def _check_multi_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_test_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.multi_validation_file))