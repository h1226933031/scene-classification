import os
from PIL import Image
import torch
from sklearn.model_selection import train_test_split


class Dataset_scene(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data_list = data

    def __getitem__(self, index):
        path_img, label = self.data_list[index]
        img = Image.open(path_img).convert('L')
        return torch.Tensor(img), label  # array to tensor?

    def __len__(self):
        return len(self.data_list)


def read_data(root, label_dic, val_ratio, seed):
    train, val = [], []
    for category in label_dic.keys():
        path_list = os.listdir(os.path.join(root, category))
        category_X = [path for path in path_list]
        train_x, val_x = train_test_split(category_X, test_size=val_ratio, random_state=seed)
        train += [(x, label_dic[category]) for x in train_x]
        val += [(x, label_dic[category]) for x in val_x]
    return train, val
