"""
Guideline of your submission of HW3.
If you have any questions in regard to submission,
please contact TA: Zheng Huan <huan_zheng@u.nus.edu>
"""

###################################### Subroutines #####################################################################
"""
Example of subroutines you might need. 
You could add/modify your subroutines in this section. You can also delete the unnecessary functions.
It is encouraging but not necessary to name your subroutines as these examples. 
"""
import os
import torch
import torch.nn as nn
import numpy as np
from utils.load_data import read_data, Dataset_scene
from model.CNN import CNN_3_layers
from model.resnet import ResNet, BasicBlock
from model.attention import Attention
from model.vgg import VGG
from utils.train import train_one_epoch, vali, train_results_plot
from torchmetrics import F1Score


def build_vocabulary(**kwargs):
    pass


def get_hist(**kwargs):
    pass


def classifier(**kwargs):
    pass


def get_accuracy(**kwargs):
    pass


def save_model(**kwargs):
    pass


###################################### Main train and test Function ####################################################
"""
Main train and test function. You could call the subroutines from the `Subroutines` sections. Please kindly follow the 
train and test guideline.

`train` function should contain operations including loading training images, computing features, constructing models, training 
the models, computing accuracy, saving the trained model, etc
`test` function should contain operations including loading test images, loading pre-trained model, doing the test, 
computing accuracy, etc.
"""


def train(train_data_dir, model_dir, args):
    """Main training model.

    Arguments:
        train_data_dir (str):   The directory of training data
        model_dir (str):        The directory of the saved model.
        args:                   The training configs and model-specified hyper-parameters.
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        train_accuracy (float): The training accuracy.
    """
    ################ 1. loading training images ################

    fix_seed = 666
    torch.manual_seed(fix_seed)

    train_data, val_data = read_data(train_data_dir, args.labels, val_ratio=args.val_ratio,
                                     seed=fix_seed, augment=args.data_augmentation)
    print('#(train_data):', len(train_data))
    print('#(val_data):', len(val_data))

    train_set = Dataset_scene(train_data, augment=args.data_augmentation)
    val_set = Dataset_scene(val_data, augment=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)

    ################ 2. enable devices ################
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Use GPU: cuda:{device}')
    else:
        device = torch.device("cpu")
        print('Use CPU')

    ################ 3. constructing models and training config ################

    model_dict = {'CNN_3_layers': CNN_3_layers, 'ResNet': ResNet, 'Attention:': Attention, 'VGG': VGG}
    model = model_dict[args.model_name](args)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    ################ 4. training & saving the models ################

    patient_count = 0
    best_val_loss = np.Inf

    total_train_loss, total_valid_loss, total_train_acc, total_valid_acc = [], [], [], []
    total_train_f1, total_val_f1 = [], []
    f1 = F1Score(task='multiclass', num_classes=args.class_num, average='macro')
    for path in [model_dir, args.fig_path]:
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
            best_val_loss, patient_count = val_loss, 0

            torch.save(model, os.path.join(model_dir, args.model_name) + '.pth')
            # torch.save(model.state_dict(), os.path.join(model_dir, args.model_name)+'.pt')
            # model.load_state_dict(torch.load(best_model_path))
        else:
            patient_count += 1

        # apply early stopping
        if args.early_stopping_patience and patient_count > args.early_stopping_patience:
            break

        scheduler.step()  # adjust learning rate

    ################ 5. evaluating the final model ################

    best_model = torch.load(os.path.join(args.model_dir, args.model_name) + '.pth')
    best_val_loss, best_val_acc, best_val_f1 = vali(best_model, f1, val_loader, criterion, args.labels, device)
    print(
        f"Evaluate the Best model: Val_Loss: {best_val_loss:.4f} -- Val_acc:{best_val_acc:.2%} -- Val_F1:{best_val_f1:.2f}")

    ################ 6. visualizing training results ################

    train_results_plot(args.model_name, total_train_loss, total_valid_loss, total_train_acc, total_valid_acc,
                       total_train_f1, total_val_f1, args.fig_path)

    ################ 7. return training accuracy ################

    if args.early_stopping_patience and patient_count > args.early_stopping_patience:
        return float(total_train_acc[-patient_count])
    return float(total_train_acc[-1])


def test1(test_data_dir, model_dir, args):
    """Main testing model.

    Arguments:
        test_data_dir (str):    The `test_data_dir` is blind to you. But this directory will have the same folder structure as the `train_data_dir`.
                                You could reuse the snippets of loading data in `train` function
        model_dir (str):        The directory of the saved model. You should load your pretrained model for testing
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        test_accuracy (float): The testing accuracy.
    """

    test_list = read_data(test_data_dir, args.labels, val_ratio=None, seed=None, test=True)
    test_loader = torch.utils.data.DataLoader(dataset=Dataset_scene(test_list), batch_size=len(test_list),
                                              shuffle=False)
    model = torch.load(model_dir, map_location=torch.device('cpu'))  # must be .pth format
    model.eval()  # disable Batch Normalization & Dropout
    k = 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = torch.argmax(model(x), dim=1)
            correct = preds.eq(y.view_as(preds)).sum()
            k += 1
        print(f'testing: {y.shape[0]} test samples in total.')
    return float(correct.int() / y.shape[0])  # test accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train', 'test'])
    parser.add_argument('--train_data_dir', default='./data/train/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./data/test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='./ckpt/', help='the pre-trained model')  # ckpt_path
    parser.add_argument('--model_name', type=str, default='CNN_3_layers', help='model name')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='the ratio of validation set')
    parser.add_argument('--class_num', type=int, default=100, help='num of image class')
    parser.add_argument('--data_augmentation', type=bool, default=True, help='whether apply data augmentation')
    # training parameters
    parser.add_argument('--epochs', type=int, default=1, help='max training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--fig_path', type=str, default='./fig/', help='path for saving visualized training results')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='early_stopping_patience')

    # model-specified parameters
    # 1. ResNet
    parser.add_argument('--resnet_input_channels', default=None, help='input_channels for ResNet')
    parser.add_argument('--resnet_output_sizes', default=None, help='output_sizes for ResNet')
    parser.add_argument('--resnet_kernel_sizes', default=None, help='kernel_sizes for ResNet')
    # 2. Attention
    parser.add_argument('--block_num', default=[1, 1, 1], help='kernel_sizes for ResNet')
    # 3. VGG
    parser.add_argument('--vgg_version', type=str, default='Modified', help="vgg version, choose from ['Modified', "
                                                                            "'A', 'B', 'D', 'E']")

    opt = parser.parse_args()
    labels = {'bedroom': 1, 'Coast': 2, 'Forest': 3, 'Highway': 4, 'industrial': 5, 'Insidecity': 6,
              'kitchen': 7, 'livingroom': 8, 'Mountain': 9, 'Office': 10, 'OpenCountry': 11, 'store': 12,
              'Street': 13, 'Suburb': 14, 'TallBuilding': 15}
    opt.labels = dict([(k.lower(), v - 1) for k, v in labels.items()])
    opt.best_model_dir = './ckpt/ResNet_b64_lr0.001_val0.1.pth'  # the final model directory

    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir, opt)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test1(opt.test_data_dir, opt.best_model_dir, opt)
        print(testing_accuracy)
