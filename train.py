from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from ipt import ImageProcessingTransformer
from functools import partial
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='Dense_Haze_NTIRE19')
parser.add_argument('--pretrained_model', type=str, default='model_1016_0.2')
parser.add_argument('--epoch', type=int, default=300)

args = parser.parse_args()
data_path = args.data_path
pretrained_model = args.pretrained_model
nb_epochs = args.epoch


class ImageProcessDataset(Dataset):
    def __init__(self, data_dir, transform):

        # split data_dir to train, train_label
        train_path = data_dir + '/' + 'hazy'
        label_path = data_dir + '/' + 'target'

        try:
            train_list = sorted(os.listdir(train_path))
            label_list = sorted(os.listdir(label_path))
        except:
            raise ValueError

        train_list = [data_dir + '/' + 'hazy' + '/' + i for i in train_list]
        label_list = [data_dir + '/' + 'target' + '/' + i for i in label_list]

        self.train_list = train_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        train_image = Image.open(self.train_list[idx])
        train_image = self.transform(train_image)

        label_image = Image.open(self.label_list[idx])
        label_image = self.transform(label_image)
        return train_image, label_image


def make_loaders(data_path):
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((128, 128)),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                ])
    train_path = os.path.join(data_path, 'train')
    valid_path = os.path.join(data_path, 'valid')
    test_path = os.path.join(data_path, 'test')
    train_dataset = ImageProcessDataset(train_path, transform=trans)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=16)
    valid_dataset = ImageProcessDataset(valid_path, transform=trans)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=16)
    test_dataset = ImageProcessDataset(test_path, transform=trans)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=16)

    return train_loader, valid_loader, test_loader

def make_model(pretrained_model):
    
    if pretrained_model == '' or pretrained_model == None:
        model = ImageProcessingTransformer(
            patch_size=4, depth=6, num_heads=4, ffn_ratio=4, qkv_bias=True,drop_rate=0.2, attn_drop_rate=0.2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), ).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) 
    
    else:
        model = ImageProcessingTransformer(
            patch_size=4, depth=6, num_heads=4, ffn_ratio=4, qkv_bias=True,drop_rate=0.2, attn_drop_rate=0.2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), ).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) 

        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

train_loader, valid_loader, test_loader = make_loaders(data_path)
model, optimizer = make_model(pretrained_model)

if pretrained_model == '' or pretrained_model == None:
    model_name = 'raw_model'
else:
    model_name = pretrained_model
