from load_data import read_data, Dataset_scene
import torch
import torch.nn as nn
from model.CNN import CNN_3_layers
from train import train_one_epoch, vali
import numpy as np


def main():
    fix_seed = 666
    torch.manual_seed(fix_seed)

    labels = {'bedroom': 1, 'Coast': 2, 'Forest': 3, 'Highway': 4, 'industrial': 5, 'Insidecity': 6,
              'kitchen': 7, 'livingroom': 8, 'Mountain': 9, 'Office': 10, 'OpenCountry': 11, 'store': 12,
              'Street': 13, 'Suburb': 14, 'TallBuilding': 15}
    labels = dict([(k, v - 1) for k, v in labels.items()])

    data_path = './train/'
    MAX_EPOCH = 1
    BATCH_SIZE = 128
    LR = 0.005
    log_interval = 10

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

    train_set = Dataset_scene(train_data)
    val_set = Dataset_scene(val_data)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

    # for train_x, train_y in train_loader:
    #     print('train_x shape', train_x.shape)
    #     print('train_y shape', train_y.shape)
    #     break
    model = CNN_3_layers().to(device)
    # model.initialize_weights()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(MAX_EPOCH):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = vali(model, val_loader, criterion, labels, device)

        print(f"Training:Epoch[{epoch:0>3}/{MAX_EPOCH:0>3}] Loss: {train_loss:.4f} Acc:{train_acc:.2%}, "
              f"Val_Loss: {val_loss:.4f} Val_:{val_acc:.2%}")

        scheduler.step()  # adjust learning rate


if __name__ == '__main__':
    main()
