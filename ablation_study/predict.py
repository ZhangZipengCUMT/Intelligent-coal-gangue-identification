import warnings
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from Fusion_Matrix import Get_Fusion_Matrix
from dataset_vib import CustomDataset
warnings.filterwarnings("ignore")
from Model_abl_3 import Proposed_model_v3
from config import AffnetConfig, RetNet_param
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from einops import repeat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def calc(TN, FP, FN, TP):
    SN = TP / (TP + FN)  # recall
    SP = TN / (TN + FP)
    Precision = TP / (TP + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    F1 = (2 * TP) / (2 * TP + FP + FN)
    fz = TP * TN - FP * FN
    fm = (TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)
    MCC = fz / pow(fm, 0.5)
    return SN, SP,Precision, ACC, MCC
batch_size = 16
aff_config = AffnetConfig()
num_epochs = 100
data_dir='Z:\数据0.2\数据集0.2'
test_dataset = CustomDataset(data_dir, 'test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
model = Proposed_model_v3(nums=2, RetNet_param=RetNet_param,
                           aff_config=aff_config,
                           feature_dim=32, num_head=8, cls_midch=2,
                           cls_lnum=1, cls_linearfeas=504, class_num=2, N_nums=2)
model.to(device)
checkpoint = torch.load(r'Z:\消融\weight_abl3\3_th try\epoch14.pth')
model.load_state_dict(checkpoint)
num_classes = 2
# df = pd.DataFrame(
#     columns=[ "Test Accuracy", "F1 Score", "SN", "SP", 'p', "MCC"])
predicted_labels=[]
true_labels=[]
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs.squeeze(1).permute(0, 2, 1).float())
        _, predicted = torch.max(outputs.data, 1)
        # print("labels.shape:", labels.shape)
        # print("predicted.shape:", predicted.shape)

        total += labels.size(0)

        correct += (predicted == labels.long()).sum().item()
        # print(correct)
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.long().cpu().numpy())

cm = confusion_matrix(true_labels, predicted_labels)

tp, fp, fn, tn = cm.ravel()
sn, sp, p, acc, mcc = calc(tn, fp, fn, tp)
# 计算 F1 分数
f1 = f1_score(true_labels, predicted_labels)
accuracy = correct / total
# print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("f1_score", f1)
print(accuracy)
print("sn=", sn, "sp=", sp, 'p=', p, "acc=", acc, "mcc=", mcc)
#
#
# new_row = { "Test Accuracy": acc, "F1 Score": f1, "SN": sn, "SP": sp, "p": p, "MCC": mcc}
# # df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
# # df.to_excel("mb2_2_8.xlsx", index=False)
# # print('Finished Training1111')
#
save_path = 'ab3.csv'
# matrix_save_path = 'gt_Fusion_Matrix.png'
# matrix_save_path = r'F:\SCI\code_and_data\实验记录\加噪\eva\1_gt_hsnn_0dB_Fusion_Matrix.png'




#test
evaluation_indicators = []
title = ['Test Accuracy', 'F1 Score', 'SN', 'SP', 'p','MCC','P_Macro', 'P_Micro', 'R_Macro', 'R_Micro', 'F1_Macro', 'F1_Micro', 'acc', 'Cohen_Kappa', 'matth', 'tp', 'fp', 'fn', 'tn']
evaluation_indicators.append(title)

# print(true_labels)
# print(predicted_labels)

y_true = true_labels#.tolist()  # 第二列除第一行的元素
y_pred = predicted_labels#.tolist()  # 第三列除第一行的元素


Matrix, (P_Macro, P_Micro), (R_Macro, R_Micro), (F1_Macro, F1_Micro), acc, Cohen_Kappa, matth = \
        Get_Fusion_Matrix(y_true=y_true, y_pred=y_pred, Save_matrix=False )
evaluation_indicators.append([acc,f1,sn,sp,p,mcc,P_Macro, P_Micro, R_Macro, R_Micro, F1_Macro, F1_Micro, acc,
                                   Cohen_Kappa, matth, tp, fp, fn, tn])


pd.DataFrame(evaluation_indicators).to_csv(save_path)

print('done')