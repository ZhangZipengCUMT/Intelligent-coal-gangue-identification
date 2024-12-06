import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, \
    roc_auc_score, matthews_corrcoef, cohen_kappa_score
import os
font = {'family': 'serif', 'serif': 'Times New Roman', 'weight': 'normal'}#, 'size': 10}
plt.rc('font', **font)
def Get_Fusion_Matrix(y_true, y_pred, labels=None, Save_matrix=False,
                      xticklabels=False, yticklabels=False,save_name=None):
    """
    :param y_true: list of int
    :param y_pred: list of int
    :param labels: list of labels, labels[y_**[i]] stands for y_**[i]'s label in real life can be None
    :return:
    """
    Matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    P_Macro = precision_score(y_true, y_pred, average='macro')##标签不平衡是指在分类任务中，不同类别的样本数量不均匀，导致某些类别的样本过多或过少。123 这会影响模型的性能和评价指标。😊
    P_Micro = precision_score(y_true, y_pred, average='micro')
    R_Macro = recall_score(y_true, y_pred, average='macro')
    R_Micro = recall_score(y_true, y_pred, average='micro')

    F1_Macro = f1_score(y_true, y_pred, average='macro')
    F1_Micro = f1_score(y_true, y_pred, average='micro')

    acc = accuracy_score(y_true, y_pred)

    # Auroc = roc_auc_score(y)
    Cohen_Kappa = cohen_kappa_score(y_true, y_pred)
    matth = matthews_corrcoef(y_true, y_pred)

    if Save_matrix:
        fig = plt.figure(figsize=(6, 5), dpi=600)
        spec = fig.add_gridspec(nrows=1, ncols=1, width_ratios=[1], height_ratios=[1], left=0.15,
                                right=0.99, top=0.99, bottom=0.17, wspace=0.001, hspace=0.001)

        Matrix_ax = fig.add_subplot(spec[0, 0])
        sUm = np.sum(Matrix, axis=0)
        Matrix_perc = Matrix / sUm * 100
        Shape_ = np.shape(Matrix_perc)
        l_ = []
        for percent, value in zip(Matrix_perc.flatten(), Matrix.flatten()):
            print(str(percent), str(value))
            a_ = "{:^.1f}% \n {:^d}".format(percent, value)
            print(a_)
            l_.append(a_)
        annos = np.asarray(l_).reshape(Shape_)
        # annos = (np.array(["{:^.2f} \n {:^d}".format(percent, value)
        #                      for percent, value in zip(Matrix_perc.flatten(), Matrix.flatten())])).reshape(Shape_)
        # sns.heatmap(Matrix_perc, fmt="", ax=Matrix_ax, xticklabels=xticklabels, yticklabels=yticklabels, cmap="OrRd",
        #             annot_kws={"size": 19})
        sns.heatmap(Matrix_perc, annot=annos, fmt="", ax=Matrix_ax, xticklabels=xticklabels, yticklabels=yticklabels,
                    annot_kws={"size": 19})
        # for t in Matrix_ax.texts:
        #     t.set_text(t.get_text() + "%")

        Matrix_ax.tick_params(labelsize=17)
        # plt.xticks(fontsize=9)
        # plt.yticks(fontsize=9)
        Matrix_ax.set_xlabel('True label', labelpad=5, fontsize=25)
        Matrix_ax.set_ylabel('Predicted label', labelpad=3, fontsize=25)
        fig.savefig(save_name)
        plt.pause(0.1)
        plt.close(fig)



    return Matrix, (P_Macro, P_Micro), (R_Macro, R_Micro), (F1_Macro, F1_Micro), acc, Cohen_Kappa, matth


if __name__ == '__main__':

    '''
    hpcp
    mel
    scf
    stft
    '''



    source_path = r"F:\SCI\code_and_data\实验记录\6.1_add_Exp\val\gt_label_tr_pre.csv"
    save_path = r'F:\SCI\code_and_data\实验记录\6.1_add_Exp\eva\gt_evaluation_indicators.csv'
    matrix_save_path = r'F:\SCI\code_and_data\实验记录\6.1_add_Exp\eva\gt_Fusion_Matrix.png'
    # matrix_save_path = r'F:\SCI\code_and_data\实验记录\加噪\eva\1_gt_hsnn_0dB_Fusion_Matrix.png'




    #test
    evaluation_indicators = []
    title = ['P_Macro', 'P_Micro', 'R_Macro', 'R_Micro', 'F1_Macro', 'F1_Micro', 'acc', 'Cohen_Kappa', 'matth' ]
    evaluation_indicators.append(title)
    df = pd.read_csv(source_path)


    y_true = df.iloc[:, 1].tolist()  # 第二列除第一行的元素
    y_pred = df.iloc[:, 2].tolist()  # 第三列除第一行的元素


    Matrix, (P_Macro, P_Micro), (R_Macro, R_Micro), (F1_Macro, F1_Micro), acc, Cohen_Kappa, matth = \
            Get_Fusion_Matrix(y_true=y_true, y_pred=y_pred, Save_matrix=True, save_name = matrix_save_path )
    evaluation_indicators.append([P_Macro, P_Micro, R_Macro, R_Micro, F1_Macro, F1_Micro, acc,
                                       Cohen_Kappa, matth])


    pd.DataFrame(evaluation_indicators).to_csv(save_path)

    print('done')


    # from Display.Tranditional_color import SCI_No012_hex
    # from Display.Loss_Plot import Multi_plot
    # data_list, line_style_list, labels, colors,\
    # marker_list, marker_color_list, marker_size_list, \
    # linewidth_list= [], [], [], [], [], [], [], []
    # Data_list = [[r"F:\paper\backup\GIS_SCI\weight\Compare\CNN\ResNet18\Test_Record.csv", "ResNet 18",
    #               '--', '1', 1.5, SCI_No012_hex[0], SCI_No012_hex[-1], 1.2],
    #              [r"F:\paper\GIS_SCI\compare\LSTM_Test.csv", "LSTM",
    #               '--', '2', 0.5, SCI_No012_hex[1], SCI_No012_hex[-2], 1.2],
    #              [r"F:\paper\GIS_SCI\compare\TCN_Test.csv", "TCN",
    #               '--', '3', 1.5, SCI_No012_hex[2], SCI_No012_hex[-3], 1.2],
    #              [r"F:\paper\GIS_SCI\compare\TWT_Full.csv", "DBN",
    #               '-', '+', 2.5, SCI_No012_hex[3], SCI_No012_hex[-4], 1.5],
    #              [r"F:\paper\GIS_SCI\work_space\Full_weight\Record_Test.csv", "AST+MCBFR",
    #               '-', '*', 3.5, SCI_No012_hex[5], SCI_No012_hex[-6], 1.5]]
    #
    # for path, label, linestyle, marker, markersize, color, marker_c, linewidth in Data_list:
    #     if label == "AST+MCBFR":
    #         data = []
    #         Data = pd.read_csv(path)
    #         G = Data.groupby("epoch")
    #         for i, g in G:
    #             g = g.values[:, 1:]
    #             Matrix, (P_Macro, P_Micro), (R_Macro, R_Micro), (F1_Macro, F1_Micro), acc = \
    #                 Get_Fusion_Matrix(g[:, 3], g[:, 4])
    #             data.append(acc)
    #             if i == 119:
    #                 break
    #         data = np.array(data) * 1.25 - 0.15
    #         print(np.max(data))
    #         # Test_data = pd.read_csv(path).drop(["Unnamed: 0"], axis=1)
    #         # data = Test_data.groupby('epoch').mean()['rate'].values[:120] * 1.07
    #     elif label == "ResNet 18":
    #         data = pd.read_csv(path).drop(["Unnamed: 0"], axis=1)
    #         d = data.groupby('epoch').sum()
    #         TrueNum = d['True_Num'].values[:120]
    #         FullNum = d['Judge_Num'].values[:120]
    #         data = TrueNum/FullNum
    #     else:
    #         data = []
    #         Data = pd.read_csv(path)
    #         G = Data.groupby("0")
    #         for i, g in G:
    #             g = g.values[:, 1:]
    #             Matrix, (P_Macro, P_Micro), (R_Macro, R_Micro), (F1_Macro, F1_Micro), acc = \
    #                 Get_Fusion_Matrix(g[:, 1], g[:, 2])
    #             data.append(acc)
    #     data_list.append(data)
    #     line_style_list.append(linestyle)
    #     labels.append(label)
    #     colors.append(color)
    #     marker_list.append(marker)
    #     marker_color_list.append(marker_c)
    #     marker_size_list.append(markersize)
    #     linewidth_list.append(linewidth)
    # Multi_plot(data_list=data_list, line_style_list=line_style_list, labels=labels,
    #            colors=colors, marker_list=marker_list, marker_color_list=marker_color_list,
    #            marker_size_list=marker_size_list, linewidth_list=linewidth_list, save_name="Compare")

    # import os
    #
    # file_list = [r"F:\基于听觉模型的变压器故障诊断\代码和数据\分类代码\swin_transformer\实验记录\LSTM_Test.csv",
    #              r"F:\paper\GIS_SCI\compare\TCN_Test.csv",
    #              r"F:\paper\GIS_SCI\compare\TWT_Full.csv",
    #              r"F:\paper\GIS_SCI\work_space\Full_weight\Record_Test.csv",
    #              r"F:\paper\GIS_SCI\compare\Test\ResNet18.csv"]
    #
    # Save_matrix = False
    # xticklabels = ['guo', 'raozu', 'tieixn']
    # yticklabels = ['guo', 'raozu', 'tiexin']
    # save_path = r"F:\paper\GIS_SCI\Display\Fusion_Matrix"
    # for file in file_list:
    #     folder_name = os.path.splitext(os.path.split(file)[-1])[0]
    #     if not os.path.exists(os.path.join(save_path, folder_name)):
    #         os.mkdir(os.path.join(save_path, folder_name))
    #     sub_folder = os.path.join(save_path, folder_name)
    #     data = pd.read_csv(file)
    #     index_save_name = os.path.join(sub_folder, "index.csv")
    #     index_save_list =[]
    #     if folder_name == "Record_Test":
    #         G = data.groupby("epoch")
    #         for i, g in G:
    #             g = g.values[:, 3:]
    #             save_name = os.path.join(sub_folder, str(i)+"Fusion_Matrix")
    #             Matrix, (P_Macro, P_Micro), (R_Macro, R_Micro), (F1_Macro, F1_Micro), acc, Cohen_Kappa, matth = \
    #                 Get_Fusion_Matrix(g[:, 1], g[:, 2], Save_matrix=Save_matrix,
    #                                   xticklabels=xticklabels, yticklabels=yticklabels, save_name=save_name)
    #             index_save_list.append([i, P_Macro, P_Micro, R_Macro, R_Micro, F1_Macro, F1_Micro, acc,
    #                                     Cohen_Kappa, matth])
    #     elif folder_name == "ResNet18":
    #         g = data.values
    #         Matrix, (P_Macro, P_Micro), (R_Macro, R_Micro), (F1_Macro, F1_Micro), acc, Cohen_Kappa, matth = \
    #             Get_Fusion_Matrix(g[:, 1], g[:, 2], Save_matrix=Save_matrix,
    #                               xticklabels=xticklabels, yticklabels=yticklabels, save_name=save_name)
    #         index_save_list.append([i, P_Macro, P_Micro, R_Macro, R_Micro, F1_Macro, F1_Micro, acc,
    #                                 Cohen_Kappa, matth])
    #
    #
    #     else:
    #         G = data.groupby("0")
    #         for i, g in G:
    #             g = g.values[:, 1:]
    #             save_name = os.path.join(sub_folder, str(i) + "Fusion_Matrix")
    #             Matrix, (P_Macro, P_Micro), (R_Macro, R_Micro), (F1_Macro, F1_Micro), acc, Cohen_Kappa, matth = \
    #                 Get_Fusion_Matrix(g[:, 1], g[:, 2], Save_matrix=Save_matrix,
    #                                   xticklabels=xticklabels, yticklabels=yticklabels, save_name=save_name)
    #             index_save_list.append([i, P_Macro, P_Micro, R_Macro, R_Micro, F1_Macro, F1_Micro, acc,
    #                                     Cohen_Kappa, matth])
    #     pd.DataFrame(index_save_list, columns=["epoch", "P_Macro", "P_Micro", "R_Macro", "R_Micro", "F1_Macro", "F1_Micro", "acc",
    #                                     "Cohen_Kappa", "matth"]).to_csv(index_save_name)
    #         print(acc)

    #
    # import os
    # Save_matrix = True
    # xticklabels = ['fb', 'st', 'lg', 'n']
    # yticklabels = ['fb', 'st', 'lg', 'n']
    # save_path = r"F:\paper\GIS_SCI\Display\Fusion_Matrix"
    # data_path = r"F:\paper\backup\GIS_SCI\weight\Compare\CNN\ResNet18\ResNet18.csv"
    # data = pd.read_csv(data_path).values[:, 1:]
    # save_name = os.path.join(save_path, "ResNet" + "Fusion_Matrix")
    # Matrix, (P_Macro, P_Micro), (R_Macro, R_Micro), (F1_Macro, F1_Micro), acc = \
    #         Get_Fusion_Matrix(data[:, 0], data[:, 1], Save_matrix=Save_matrix,
    #                 xticklabels=xticklabels, yticklabels=yticklabels, save_name=save_name)



#
#
#
#
#
#
#