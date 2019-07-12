import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import os
import copy
import argparse
import time
import re
import numpy as np
import pandas as pd

def make_dataset(dir, phase, label_types):
    images = []
    labels = []
    path = os.path.join(dir, phase)
    files = os.listdir(path)
    for i in range(len(label_types)):
        label = label_types[i]
        image_dir_path = os.path.join(path, label)
        image_files = os.listdir(image_dir_path)
        jpeg_files = [f for f in image_files if re.match('.*jpeg', f)]
        for jpeg_path in jpeg_files:
            image_path = os.path.join(image_dir_path, jpeg_path)
            # one hot vector でクラスを表す
            y = [0, 0]
            y[i] = 1
            images.append(image_path)
            labels.append(np.array(y))
    return images, labels

class DatasetFolder(data.Dataset):
    def __init__(self, X, y, loader, transform=None, target_transform=None):
        self.loader = loader
        self.samples = X
        self.targets = y
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path = self.samples[index]
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# クラスの変換。今回はPytorchのTensor型に変換するだけ
# 他に必要な変換がある場合には、画像同様に記載可能。
class ToTensorOfTarget(object):
    def __call__(self, target):
        return torch.from_numpy(target)

def train_model(device, dataloaders, dataset_sizes, 
                model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # 途中経過でモデル保存するための初期化
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000.0
    # 時間計測用
    end = time.time()

    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        print('Epoch:{}/{}'.format(epoch, num_epochs - 1), end="")

        # 各エポックで訓練+バリデーションを実行
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                labels = labels.float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 訓練のときだけ履歴を保持する
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, classnums = torch.max(labels, 1)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, classnums)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 統計情報
                running_loss += loss.item() * inputs.size(0)
                running_corrects += float(torch.sum(preds == classnums))

            # サンプル数で割って平均を求める
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('\t{} Loss: {:.4f} Acc: {:.4f} Time: {:.4f}'.format(phase, epoch_loss, epoch_acc, time.time()-end), end="")
            if phase == 'train':
                train_loss_list.append(epoch_loss)
            else:
                val_loss_list.append(epoch_loss)

            # 精度が改善したらモデルを保存する
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            end = time.time()

        print()

    time_elapsed = time.time() - since
    print()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:.4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)    
    return model, train_loss_list, val_loss_list

def calc_test_accuracy(device, dataloader, dataset_size, model, criterion):
    running_loss = 0.0
    running_corrects = 0
    model.train(False)
    y_pred = []
    y_true = []

    for inputs, labels in dataloader:
        labels = labels.float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, classnums = torch.max(labels, 1)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, classnums)

        # 統計情報
        running_loss += loss.item() * inputs.size(0)
        running_corrects += float(torch.sum(preds == classnums))
        # 精度の計算
        y_pred = y_pred + list(preds.cpu().numpy())
        y_true = y_true + list(classnums.cpu().numpy())

    # サンプル数で割って平均を求める
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    print('On Test:\tLoss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return y_true, y_pred

def train_and_test(datadir, label_set, params):
    batch_size = params.get("batch_size", 64)
    epochs = params.get("epochs", 5)
    lr = params.get("lr", 0.0001)
    momentum = params.get("momentum", 0.90)
    step_size = params.get("step_size", 5)
    gamma = params.get("gamma", 0.1)
    pretrained = params.get("pre_trained", True)
    train_mode = params.get("train_mode", "FT") # FT: fine-tuning, TF: transfar-learning

    # 変換後の画像の幅と高さ
    WIDTH = 224
    HEIGHT = 224

    if torch.cuda.is_available(): # GPUが利用可能か確認
        device = 'cuda'
    else:
        device = 'cpu'

    print("Settings:")
    print("\tDevice:", device)
    print("\tBatch size:", batch_size)
    print("\tEpochs:", epochs)
    print("\tLearning rate:",  lr)
    print("\tMomentum(SGD):", momentum)
    print("\tStep size for LR:", step_size)
    print("\tGamma for LR:", gamma)
    if pretrained:
        print("\tPretrained model: Use")
        if train_mode == "FT":
            print("\t\tTrain Mode: Fine Tuning")
        elif train_mode == "TL":
            print("\t\tTrain Mode: Transfer Learning")
    else:
        print("\tPretrained model: Not use")
    print()


    X_train, y_train = make_dataset(datadir, "train", label_set)
    X_val, y_val = make_dataset(datadir, "val", label_set)
    X_test, y_test = make_dataset(datadir, "test", label_set)

    print("# of samples:")
    print("\tTraining: {:d}".format(len(X_train)))
    print("\tValidation: {:d}".format(len(X_val)))
    print("\tTest: {:d}".format(len(X_test)))
    print()

    # 画像の輝度値を補正するための関数を設定
    # ResNet等のPre-trained model 学習時に利用されていた値を利用
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])


    # training (validation, testも同様）時に、画像に対して変換を加える場合は、
    # ここに記述する。ResizeやFlipなど。
    # 参照：https://pytorch.org/docs/stable/torchvision/transforms.html
    # 変換のあと、pytorchで扱うために、Tensor型に変換してあげる必要あり。
    # normalize(上記の関数)は、Tensor型に変換したあと、実施
    data_transforms = {
        # training data用。必要ならaugmentation(Flipや切り出し)を行う
        # 今は、特段の加工は行わない。
        'train': transforms.Compose([
            transforms.Resize((WIDTH, HEIGHT)),
            transforms.ToTensor(),
            normalize
        ]),
        # validation用。通常はFlip等は行わない。
        'val': transforms.Compose([
            transforms.Resize((WIDTH, HEIGHT)),
            transforms.ToTensor(),
            normalize
        ]),
        # test用。こちらもFlip等は実施しない
        'test': transforms.Compose([
            transforms.Resize((WIDTH, HEIGHT)),
            transforms.ToTensor(),
            normalize
        ])
    }

    target_transforms = transforms.Compose([
        ToTensorOfTarget()
    ])

    # 画像とクラスの読み込み用の関数を定義
    image_datasets = {
        'train':DatasetFolder(X_train, y_train, pil_loader,
                                data_transforms['train'],
                                target_transforms),
        'val':DatasetFolder(X_val, y_val, pil_loader,
                                data_transforms['val'],
                                target_transforms),
        'test': DatasetFolder(X_test, y_test, pil_loader,
                                data_transforms['test'],
                                target_transforms)
    }

    # バッチサイズ分のデータを読み込む。
    # training はデータをシャッフルし、読み込む画像の順番をランダムにする。
    # 他はシャッフルの必要なし。
    workers=0
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers),
        'val': torch.utils.data.DataLoader(
            image_datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers),
        'test': torch.utils.data.DataLoader(
            image_datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers)
    }

    # 訓練, 評価, テストの画像の大きさをカウントする。
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    # モデルの初期化
    model = models.resnet18(pretrained=pretrained)
    if train_mode == 'TL':
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    # 損失関数、
    # パラメータの最適化方法、学習率の更新方法を定義。
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 実際の学習を実施
    model, train_loss_list, val_loss_list = train_model(device, dataloaders, dataset_sizes, model, 
                                                        criterion, optimizer, exp_lr_scheduler, num_epochs=epochs)

    # テストデータでの精度を求める
    y_true, y_pred = calc_test_accuracy(device, dataloaders['test'], dataset_sizes['test'], model, criterion)
    pd.DataFrame(confusion_matrix(y_true, y_pred), columns=label_set)
    
    return model, train_loss_list, val_loss_list

if __name__ == "__main__":
    datadir = "chest_xray_exe"
    label_set = ["NORMAL", "PNEUMONIA"]
    params = {
        "epochs":10,
        "batch_size":64,
        "lr":0.0005,
        "momentum":0.95,
        "pretrained":True,
        "train_mode":"FT"
    }
    final_model, train_loss, val_loss = train_and_test(datadir, label_set, params)

    model_file_name = "best_model.torch"
    torch.save(final_model.state_dict(), model_file_name)

    p1 = plt.plot(list(range(len(train_loss))), train_loss, linestyle="dashed")
    p2 = plt.plot(list(range(len(val_loss))), val_loss, linestyle="solid")
    plt.legend((p1[0], p2[0]), ("Training", "Validation"), loc=1)
