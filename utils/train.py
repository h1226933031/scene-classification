import torch
import os
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def compute_acc_n_f1(preds, y, f1):
    """
    Returns accuracy and f1 score for the train/val dataset, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # top_pred = torch.argmax(preds, dim=1)
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.int() / y.shape[0]
    return acc, f1(preds, y)


# def categorical_accurate(preds, y, start_label_index=0):
#     """
#     Returns accuracy per batch, i.e. if you get 8/10 right, this returns 8
#     """
#     top_pred = torch.argmax(preds, dim=1).add(start_label_index)
#     correct = top_pred.eq(y.view_as(top_pred)).sum()
#     return correct.int()


def vali(model, f1, val_loader, criterion, label_dic, device):
    total_loss, total_correct, total = 0., 0, 0
    preds, trues = torch.empty(0).to(device), torch.empty(0).to(device)
    model.eval()  # disable Batch Normalization & Dropout
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total += labels.shape[0]

            preds = torch.cat((preds, torch.argmax(outputs, dim=1)), dim=0)
            trues = torch.cat((trues, labels), dim=0)

    print('validation classification reports:')
    print(classification_report(preds.cpu().numpy().tolist(), trues.cpu().numpy().tolist(), target_names=label_dic.keys()))
    acc, f1score = compute_acc_n_f1(preds.cpu(), trues.cpu(), f1)
    model.train()  # able Batch Normalization & Dropout
    return total_loss / total, acc, f1score


def train_one_epoch(model, f1, train_loader, optimizer, criterion, device):
    epoch_loss, total_correct, total = 0., 0., 0
    preds, trues = torch.empty(0).to(device), torch.empty(0).to(device)
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward
        outputs = model(inputs)  # [batch_size, 15]
        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        # update weights
        optimizer.step()

        preds = torch.cat((preds, torch.argmax(outputs, dim=1)), dim=0)
        trues = torch.cat((trues, labels), dim=0)
        epoch_loss += loss.item()
        total += labels.shape[0]

    acc, f1score = compute_acc_n_f1(preds.cpu(), trues.cpu(), f1)
    return epoch_loss / total, acc, f1score


def train_results_plot(model_name, total_train_loss, total_valid_loss, total_train_acc, total_valid_acc,
                       total_train_f1, total_val_f1, save_path):
    x_index = range(1, len(total_train_loss)+1)
    fig, ax = plt.subplots()
    ax.plot(x_index, total_train_loss, label='training loss')
    ax.plot(x_index, total_valid_loss, label='validation loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title(f'{model_name} loss fig')
    ax.legend()
    plt.savefig(os.path.join(save_path, model_name+'-loss.png'))

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].plot(x_index, total_train_acc, label='training accuracy')
    ax[0].plot(x_index, total_valid_acc, label='validation accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('accuracy')
    ax[0].set_title(f'{model_name} accuracy fig')
    ax[0].legend()

    ax[1].plot(x_index, total_train_f1, label='training f1')
    ax[1].plot(x_index, total_val_f1, label='validation f1')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('f1 score')
    ax[1].set_title(f'{model_name} f1-score fig')
    ax[1].legend()

    plt.savefig(os.path.join(save_path, model_name+'-accuracy+f1score.png'))

