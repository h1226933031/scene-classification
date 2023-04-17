import numpy as np
import os
from load_data import read_data, Dataset_scene
import torch
import torch.nn as nn
from model.CNN import CNN_3_layers
from model.resnet import ResNet
from model.attention import Attention
from model.vgg import VGG
from utils.train import train_one_epoch, vali, train_results_plot
import argparse
from torchmetrics import F1Score


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

    train_data, val_data = read_data(args.data_path, args.labels, val_ratio=args.val_ratio, seed=fix_seed)
    print('#(train_data):', len(train_data))
    print('#(val_data):', len(val_data))

    train_set = Dataset_scene(train_data, augment=args.data_augmentation)
    val_set = Dataset_scene(val_data, augment=args.data_augmentation)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)

    model_dict = {'CNN_3_layers': CNN_3_layers, 'ResNet': ResNet, 'Attention:': Attention, 'VGG': VGG}
    if args.model_name == 'VGG':
        model = VGG(args.vgg_version)
    else:
        model = model_dict[args.model_name]()
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    patient_count = 0
    best_val_loss = np.Inf

    total_train_loss, total_valid_loss, total_train_acc, total_valid_acc = [], [], [], []
    total_train_f1, total_val_f1 = [], []
    f1 = F1Score(task='multiclass', num_classes=15, average='macro').to(device)
    for path in [args.ckpt_path, args.fig_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(model, f1, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = vali(model, f1, val_loader, criterion, args.labels, device)

        total_train_loss.append(train_loss)
        total_valid_loss.append(val_loss)
        total_train_acc.append(train_acc)
        total_valid_acc.append(val_acc)
        total_train_f1.append(train_f1)
        total_val_f1.append(val_f1)

        print(f"Training:Epoch[{epoch + 1}/{args.epochs}] -- Train_Loss: {train_loss:.4f} -- Train_Acc:{train_acc:.2%} "
              f"-- Train_F1:{train_f1:.2f}; Val_Loss: {val_loss:.4f} -- Val_acc:{val_acc:.2%} -- Val_F1:{val_f1:.2f}")

        if val_loss < best_val_loss:  # update model
            patient_count = 0
            best_val_loss = val_loss
            if args.ckpt_path:
                torch.save(model.state_dict(), os.path.join(args.ckpt_path, args.model_name))
                # model.load_state_dict(torch.load(best_model_path))
        else:
            patient_count += 1

        # apply early stopping
        if args.early_stopping_patience and patient_count > args.early_stopping_patience:
            break

        scheduler.step()  # adjust learning rate

    # visualize training results
    train_results_plot(args.model_name, total_train_loss, total_valid_loss, total_train_acc, total_valid_acc,
                       total_train_f1, total_val_f1, args.fig_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN family for Scene Multi-Classification')
    # basic configs
    parser.add_argument('--model_name', type=str, default='CNN_3_layers', help='model name')
    parser.add_argument('--data_path', type=str, default='./train/', help='path of the data file')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='the ratio of validation set')
    parser.add_argument('--data_augmentation', type=bool, default=False, help='whether apply data augmentation')
    # training parameters
    parser.add_argument('--epochs', type=int, default=5, help='max training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='batch size')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/', help='path for saving the best model')
    parser.add_argument('--fig_path', type=str, default='./fig/', help='path for saving visualized training results')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='early_stopping_patience')

    args = parser.parse_args()

    labels = {'bedroom': 1, 'Coast': 2, 'Forest': 3, 'Highway': 4, 'industrial': 5, 'Insidecity': 6,
              'kitchen': 7, 'livingroom': 8, 'Mountain': 9, 'Office': 10, 'OpenCountry': 11, 'store': 12,
              'Street': 13, 'Suburb': 14, 'TallBuilding': 15}
    args.labels = dict([(k, v - 1) for k, v in labels.items()])
    args.class_num = 15
    args.attention_block_num = [1, 1, 1]
    args.vgg_version = 'Modified'
    main(args)
