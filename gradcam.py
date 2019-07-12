from torch.autograd import Variable
import cv2
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import train

class GradCam:
    def __init__(self, model):
        self.model = model.eval()
        self.feature = None
        self.gradient = None
    
    
    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, x):
        image_size = (x.size(-1), x.size(-2))
        datas = Variable(x)
        heat_maps = []
        #pred_lst = []
        scores_lst = []
        for i in range(datas.size(0)):
            img = datas[i].data.cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)

            feature = datas[i].unsqueeze(0)
            # ネットワークによって層の名前が違っているので、それに対する対応。
            for name, module in self.model.named_children():
                # VGG の場合はこちらが実行される
                if self.model.__class__.__name__ == "VGG":
                    if name == 'classifier':
                        feature = feature.view(feature.size(0), -1)
                    feature = module(feature)
                    if name == 'features':
                        feature.register_hook(self.save_gradient)
                        self.feature = feature
                # ResNet の場合はこちらが実行される
                elif self.model.__class__.__name__ == "ResNet":
                    if name == 'fc':
                        feature = feature.view(feature.size(0), -1)
                    feature = module(feature)
                    if name == 'layer4':
                        feature.register_hook(self.save_gradient)
                        self.feature = feature
            # 対象のクラスを１、それ以外を０として、バックプロパゲーションを実施する
            classes = torch.softmax(feature, dim=1)
            one_hot, class_id = classes.max(dim=-1)
            #pred_lst.append(class_id.item())
            self.model.zero_grad()
            one_hot.backward()
            
            # 得られた重みを平均化する
            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            # 入力画像の大きさに合わせる
            mask = cv2.resize(mask.data.cpu().numpy(), image_size)  
            # 0.0 - 1.0 の値にノーマライズする
            mask = mask - np.min(mask)
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))  # 入力画像に重ねる
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
            scores_lst.append(classes)
        heat_maps = torch.stack(heat_maps)
        return heat_maps, scores_lst

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

def run(model_path, datadir):
    phase = 'test'
    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
    # 変換後の画像の幅と高さ
    WIDTH = 224
    HEIGHT = 224

    if torch.cuda.is_available(): # GPUが利用可能か確認
        device = 'cuda'
    else:
        device = 'cpu'
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.train(False)
    model = model.to(device)

    X_test, y_test = train.make_dataset(datadir, "test", CLASS_NAMES)

    # 画像の輝度値を補正するための関数を設定
    # ResNet等のPre-trained model 学習時に利用されていた値を利用
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    data_transform = transforms.Compose([
        transforms.Resize((WIDTH, HEIGHT)),
        transforms.ToTensor(),
        normalize
    ])
    target_transforms = transforms.Compose([
        train.ToTensorOfTarget()
    ])
    # 画像とクラスの読み込み用の関数を定義
    image_dataset = DatasetFolder(X_test, y_test, pil_loader,
                            data_transform,
                            target_transforms)
    dataloader = torch.utils.data.DataLoader(
            image_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0)
    image_num = 30
    image_col = 5
    image_row = np.ceil(image_num/image_col)
    fig = plt.figure(figsize=(5*image_col, 8*image_row))

    # Run Grad-CAM
    grad_cam = GradCam(model)

    # 最初の3画像の予測結果を表示
    idx = 0
    for inputs, labels in dataloader:
        if idx >= image_num:
            break
        labels = labels.float()
        inputs = inputs.to(device)
        labels = labels.to(device)

        # GradCAMクラスの　__call__ を実行する
        # バッチに含まれる全てのGrad-CAMの結果が feaqture_image に, 
        # 2クラスの予測のスコアが scores_lst に保存される
        feature_image_lst, scores_lst = grad_cam(inputs)

        feature_image = transforms.ToPILImage()(feature_image_lst[0])
        test_image_pil = transforms.ToPILImage()(inputs[0].cpu())
        image_size = test_image_pil.size
        feature_image = feature_image.resize(image_size)    
        
        # 入力画像を表示
        img_org = pil_loader(X_test[idx])
        img_org = img_org.resize((WIDTH, HEIGHT))  # 224 x 224 に大きさを変更
        idx_row = 2*np.floor(idx / image_col)
        idx_col = idx % image_col
        # 正解クラスを取得
        gt_cls = CLASS_NAMES[np.argmax(labels[0].cpu())]
        plt.subplot(2*image_row, image_col, idx_row*image_col + idx_col + 1)
        plt.imshow(img_org)
        plt.title("Truth: {}".format(gt_cls))
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)

        # Grad-CAMの結果のヒートマップ画像を表示
        plt.subplot(2*image_row, image_col, (idx_row+1)*image_col + idx_col + 1)
        plt.imshow(feature_image)
        scores = scores_lst[0]
        pred_cls = CLASS_NAMES[scores.argmax()]
        plt.title("Pred: {} (Score: {:.4f})".format(
            pred_cls, scores.max().item()))
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)
        idx += 1
    plt.savefig("gradcam.jpg")

#if __name__ == "__main__":
#    main("best_model.torch")

if __name__ == "__main__":
    datadir = "chest_xray_exe"
    run("best_model.torch", datadir)

