import os, shutil
import warnings

from dataset_vib import CustomDataset

warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from einops import repeat
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score


from Model import Proposed_model_v1, Proposed_model_v3
from config import AffnetConfig, RetNet_param


brunch = 'weight'
attempt_id =20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def calc(TN, FP, FN, TP):
#     SN = TP / (TP + FN)  # recall
#     SP = TN / (TN + FP)
#     Precision = TP / (TP + FP)
#     ACC = (TP + TN) / (TP + TN + FN + FP)
#     F1 = (2 * TP) / (2 * TP + FP + FN)
#     fz = TP * TN - FP * FN
#     fm = (TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)
#     MCC = fz / pow(fm, 0.5)
#     return SN, SP, Precision, ACC, MCC

data_dir='Z:\coal_gangue_detection\数据0.2\数据集0.2'
batch_size = 16
aff_config = AffnetConfig()
num_epochs = 100
train_dataset = CustomDataset(data_dir, 'train')
test_dataset = CustomDataset(data_dir, 'test')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                          prefetch_factor=10)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)



model = Proposed_model_v3(nums=2, RetNet_param=RetNet_param,
                           aff_config=aff_config,
                           feature_dim=32, num_head=8, cls_midch=2,
                           cls_lnum=1, cls_linearfeas=504, class_num=2, N_nums=2).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.sgdm(model.parameters(), lr=2e-5)
# model.to(device)


# df = pd.DataFrame(
#     columns=["Epoch", "Loss", "Train Accuracy", "Test Accuracy", "F1 Score", "SN", "SP", 'p', "MCC"])
# predicted_labels = []
# true_labels = []
# best_acc = 0

def train(train_loader, model, criterion, optimizer):
    losses, labels_, preds_ = [], [], []
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.squeeze(1).permute(0, 2, 1).float())

        loss = criterion(outputs, labels.long())

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        losses.append(loss.item())
        predicted = predicted.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        for p_, l_ in zip(predicted, labels):
            labels_.append(l_)
            preds_.append(p_)
    labels_ = np.array(labels_)
    preds_ = np.array(preds_)
    return np.mean(losses), sum(labels_ == preds_) / len(labels_), labels_, preds_



def test(test_loader, model):
    labels_, preds_ = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.squeeze(1).permute(0, 2, 1).float())
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            for p_, l_ in zip(predicted, labels):
                labels_.append(l_)
                preds_.append(p_)
        labels_ = np.array(labels_)
        preds_ = np.array(preds_)
        print(len(labels_))
        print(sum(labels_ == preds_))
    return sum(labels_ == preds_) / len(labels_), labels_, preds_


def save_model(model, folder, epoch):
    try:
        torch.save(model.state_dict(), os.path.join(folder, "epoch" + str(epoch) + ".pth"))
        print("save successfully")
    except:
        print("fall to save @ epoch", epoch)


if __name__ == '__main__':
    his = []
    save_folder = os.path.join(brunch, str(attempt_id) + "_th try")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    shutil.copy("train0.5.py", os.path.join(save_folder, "train0.5.py"))
    shutil.copy("Model.py", os.path.join(save_folder, "Model.py"))
    shutil.copy("config.py", os.path.join(save_folder, "config.py"))
    for epoch in range(num_epochs):
        loss, acc, labels_, preds_ = train(train_loader=train_loader, model=model,
                                           criterion=criterion, optimizer=optimizer)
        print(epoch, loss, acc, "Train")
        test_acc, test_labels_, test_preds_ = test(test_loader=test_loader, model=model)
        print(epoch, test_acc, "Test")
        save_model(model, folder=save_folder, epoch=epoch)
        his.append([epoch, loss, acc, test_acc])
    pd.DataFrame(his, columns=["epoch", "loss", "acc", "test_acc"]).to_csv(os.path.join(save_folder, "Train_Nadam.csv"))

