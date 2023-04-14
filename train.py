import torch
import numpy as np
from sklearn.metrics import classification_report


def categorical_accurate(preds, y, start_label_index=0):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 8
    """
    top_pred = torch.argmax(preds, dim=1).add(start_label_index)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    return correct.int()


def vali(model, val_loader, criterion, label_dic, device):
    total_loss, total_correct, total = 0., 0, 0
    preds = []
    trues = []
    model.eval()  # disable Batch Normalization & Dropout
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.to(device))
            preds += torch.argmax(outputs, dim=1).numpy().tolist()
            trues += labels.numpy().tolist()
            # pred = output  # .detach().cpu()
            # true = labels  # .detach().cpu()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_correct += categorical_accurate(outputs, labels)
            total += labels.shape[0]
    print('validation classification reports:')
    print(classification_report(preds, trues, target_names=label_dic.keys()))
    model.train()  # able Batch Normalization & Dropout
    return total_loss / total, total_correct / total


def train_one_epoch(model, train_loader, optimizer, criterion):
    epoch_loss, total_correct, total = 0., 0., 0
    model.train()
    for inputs, labels in train_loader:
        # forward
        outputs = model(inputs)
        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        # update weights
        optimizer.step()

        total_correct += categorical_accurate(outputs, labels)
        epoch_loss += loss.item()
        total += labels.shape[0]

    return epoch_loss / total, total_correct / total
