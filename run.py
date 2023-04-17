import numpy as np
import os
from load_data import read_data, Dataset_scene
import torch
import torch.nn as nn
from model.CNN import CNN_3_layers
from model.resnet import ResNet
from model.attention import Attention
from utils.train import train_one_epoch, vali
import argparse


def main(args):
    fix_seed = 666
    torch.manual_seed(fix_seed)

    # enable devices
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Use GPU: cuda:{device}')
    else:
        device = torch.device("cpu")
        print('Use CPU')

    train_data, val_data = read_data(args.data_path, labels, val_ratio=args.val_ratio, seed=fix_seed)
    print('#(train_data):', len(train_data))
    print('#(val_data):', len(val_data))

    train_set = Dataset_scene(train_data, augment=args.data_augmentation)
    val_set = Dataset_scene(val_data, augment=args.data_augmentation)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)

    model_dict = {'CNN_3_layers': CNN_3_layers, 'ResNet': ResNet, 'Attention:': Attention}
    model = model_dict[args.model_name]()
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = np.Inf
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = vali(model, val_loader, criterion, labels, device)

        print(f"Training:Epoch[{epoch + 1}/{args.epochs}] Train_Loss: {train_loss:.4f} Train_Acc:{train_acc:.2%}, "
              f"Val_Loss: {val_loss:.4f} Val_acc:{val_acc:.2%}")
        if args.ckpt_path and val_loss < best_val_loss:  # update model
            torch.save(model.state_dict(), args.ckpt_path)
        # model.load_state_dict(torch.load(best_model_path))
        scheduler.step()  # adjust learning rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN family for Scene Multi-Classification')
    # basic configs
    parser.add_argument('--model_name', type=str, default=1, help='status')
    parser.add_argument('--data_path', type=str, default='./train/', help='path of the data file')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='the ratio of validation set')
    parser.add_argument('--data_augmentation', type=bool, default=False, help='whether apply data augmentation')
    # training parameters
    parser.add_argument('--epochs', type=int, default=1, help='max training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='batch size')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/', help='path for saving the best model')
    args = parser.parse_args()

    labels = {'bedroom': 1, 'Coast': 2, 'Forest': 3, 'Highway': 4, 'industrial': 5, 'Insidecity': 6,
              'kitchen': 7, 'livingroom': 8, 'Mountain': 9, 'Office': 10, 'OpenCountry': 11, 'store': 12,
              'Street': 13, 'Suburb': 14, 'TallBuilding': 15}
    args.labels = dict([(k, v - 1) for k, v in labels.items()])
    args.class_num = 15
    args.attention_block_num = [1, 1, 1]
    main(args)
