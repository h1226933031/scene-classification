import os
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torchvision.transforms as transforms


def read_data(root, label_dic, val_ratio, seed, augment=False):
    train, val = [], []
    for category in label_dic.keys():
        dir_list = os.listdir(os.path.join(root, category))
        category_X = [os.path.join(root, category, path) for path in dir_list]
        if augment:
            category_X += [path + '_r' for path in category_X]
        train_x, val_x = train_test_split(category_X, test_size=val_ratio, random_state=seed)
        train += [(x, label_dic[category]) for x in train_x]
        val += [(x, label_dic[category]) for x in val_x]
    return train, val


class Dataset_scene(torch.utils.data.Dataset):
    def __init__(self, data, augment=False, desired_size=224):
        self.data_list = data
        self.augment = augment
        # a series of transformations using torchvision.transforms, from PLI Image to normalized float tensors
        self.transform = transforms.Compose([transforms.Resize((desired_size, desired_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,)),
                                             ])

    def __getitem__(self, index):
        path_img, label = self.data_list[index]
        if self.augment and path_img[-2:] == '_r':  # horizontally flip the image
            img = Image.open(path_img[:-2]).convert('L')
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # H x W, reverse W index
        else:
            img = Image.open(path_img).convert('L')  # H x W

        return self.transform(img), label  # [C=1, H x W]

    def __len__(self):
        return len(self.data_list)
