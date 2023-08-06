"""
@Project: Farmer_identify
@File   : train.py
@Author : Ruiqing Tang
@Date   : 2023/8/5 17:12
"""
import time
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

epoch = 0
batch_idx = 0
best_val_accuracy = 0

warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

train_transform = transforms.Compose(
    [transforms.RandomSizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset_dir = "E:\\dataset\\farmerdata\\mydataset_split"
train_path = os.path.join(dataset_dir, "mytrain")
val_path = os.path.join(dataset_dir, 'myval')

train_dataset = datasets.ImageFolder(train_path, train_transform)
val_dataset = datasets.ImageFolder(val_path, val_transform)
class_names = train_dataset.classes
n_class = len(class_names)
train_dataset.class_to_idx
idx_to_labels = {y: x for x, y in train_dataset.class_to_idx.items()}
np.save('E:\\dataset\\farmerdata\\idx_to_labels.npy', idx_to_labels)
np.save('E:\\dataset\\farmerdata\\labels_to_idx.npy', train_dataset.class_to_idx)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 25)
optimizer = optim.Adam(model.parameters())

model = model.to(device)
criterion = nn.CrossEntropyLoss()
EPOCHS = 30
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)


def train_one_batch(images, labels):
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    log_train = {}
    log_train['epoch'] = epoch
    log_train['train_loss'] = loss
    log_train['train_accuracy'] = accuracy_score(labels, preds)
    log_train['train_f1-score'] = f1_score(labels, preds, average='macro')
    return log_train


def evaluate_valset():
    loss_list = []
    labels_list = []
    preds_list = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
    log_val = {}
    log_val['epoch'] = epoch
    log_val['val_loss'] = np.mean(loss_list)
    log_val['val_accuracy'] = accuracy_score(labels_list, preds_list)
    log_val['val_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    return log_val


df_train_log = pd.DataFrame()
log_train = {}
log_train['epoch'] = 0
log_train['batch'] = 0
images, labels = next(iter(train_loader))
log_train.update(train_one_batch(images, labels))
df_train_log = df_train_log.append((log_train), ignore_index=True)
df_val_log = pd.DataFrame()
log_val = {}
log_val['epoch'] = 0
log_val.update(evaluate_valset())
df_val_log = df_val_log.append(log_val, ignore_index=True)

import wandb

wandb.init(project='Farmer_identify', name=time.strftime('%m%d%H%M%S'))

for epoch in range(1, EPOCHS + 1):
    print((f'Epoch {epoch}/{EPOCHS}'))

    model.train()
    for images, labels in tqdm(train_loader):
        batch_idx += 1
        log_train = train_one_batch(images, labels)
        df_train_log = df_train_log.append(log_train, ignore_index=True)
        wandb.log(log_train)

    lr_scheduler.step()

    model.eval()
    log_val = evaluate_valset()
    df_val_log = df_val_log.append(log_val, ignore_index=True)
    wandb.log(log_val)

    if log_val['val_accuracy'] > best_val_accuracy:
        old_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(best_val_accuracy)
        if os.path.exists(old_best_checkpoint_path):
            os.remove(old_best_checkpoint_path)
        best_val_accuracy = log_val['val_accuracy']
        new_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(log_val['val_accuracy'])
        torch.save(model, new_best_checkpoint_path)
        print('new_best_model', 'checkpoint/best-{:.3f}.pth'.format(best_val_accuracy))

df_train_log.to_csv('train_log_trainset.csv', index=False)
df_val_log.to_csv('train_log_valset.csv', index=False)
model = model = torch.load('checkpoint/best-{:.3f}.pth'.format(best_val_accuracy))
# model = model = torch.load('checkpoint/best-0.893.pth')
model.eval()
print(evaluate_valset())
