import numpy as np
import os
from load_data import read_data, Dataset_scene
import torch
import torch.nn as nn
from model.CNN import CNN_3_layers
from utils.train import train_one_epoch, vali


def main():
    fix_seed = 666
    torch.manual_seed(fix_seed)

    # dataset parameters
    data_path = './train/'
    augment = True
    labels = {'bedroom': 1, 'Coast': 2, 'Forest': 3, 'Highway': 4, 'industrial': 5, 'Insidecity': 6,
              'kitchen': 7, 'livingroom': 8, 'Mountain': 9, 'Office': 10, 'OpenCountry': 11, 'store': 12,
              'Street': 13, 'Suburb': 14, 'TallBuilding': 15}
    labels = dict([(k, v - 1) for k, v in labels.items()])

    # training parameters
    MAX_EPOCH = 5
    BATCH_SIZE = 128
    LR = 0.01
    best_model_path = './ckpt/'

    # enable devices
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Use GPU: cuda:{device}')
    else:
        device = torch.device("cpu")
        print('Use CPU')

    train_data, val_data = read_data(data_path, labels, val_ratio=0.2, seed=fix_seed)
    print('#(train_data):', len(train_data))
    print('#(val_data):', len(val_data))

    train_set = Dataset_scene(train_data, augment=augment)
    val_set = Dataset_scene(val_data, augment=augment)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN_3_layers().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = np.Inf
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    for epoch in range(MAX_EPOCH):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = vali(model, val_loader, criterion, labels, device)

        print(f"Training:Epoch[{epoch + 1}/{MAX_EPOCH}] Train_Loss: {train_loss:.4f} Train_Acc:{train_acc:.2%}, "
              f"Val_Loss: {val_loss:.4f} Val_acc:{val_acc:.2%}")
        if val_loss < best_val_loss:  # update model
            torch.save(model.state_dict(), best_model_path)
        # model.load_state_dict(torch.load(best_model_path))
        scheduler.step()  # adjust learning rate


if __name__ == '__main__':
    main()
