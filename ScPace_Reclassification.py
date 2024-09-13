from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import torch
from sklearn.metrics import hinge_loss
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score


def scpace(data, data_test, labels, labels_test, C, num_iteration, p, reduce,data_no_moving,labels_no_moving):
    est = SVC(C=1, kernel='rbf', class_weight='balanced')
    C = C
    pca = PCA(n_components=0.7)
    from sklearn.decomposition import KernelPCA
    if reduce == None:
        data_no_moving_scpace = data
    if reduce == "pca":
        pca.fit_transform(data)
        d = pca.n_components_
        real_m = 0
        if d < 10:
            real_m = 10
        if 10 <= d <= 20:
            real_m = d
        if d > 20:
            real_m = 20

        pca2 = PCA(n_components=real_m)

        data = pca2.fit_transform(data)
        data_test = pca2.transform(data_test)
        data_no_moving_scpace = data

    if reduce == "kernel":

        pca2 = KernelPCA(n_components=20)

        data = pca2.fit_transform(data)
        data_test = pca2.transform(data_test)
        data_no_moving_scpace = data

    score_max_list = []
    for i in range(num_iteration):
        C = C + p
        est.C = C

        est.fit(data, labels)
        score_training = balanced_accuracy_score(est.predict(data), labels)

        pred_decision = est.decision_function(data_no_moving_scpace)
        y_true = labels_no_moving
        loss = hinge_loss(y_true, pred_decision, labels=np.array([0, 1, 2, 3]))

        def convert_v(single_loss, threshold):
            v_list = single_loss < threshold

            def convert(x: list):
                v_list = []
                for i in x:
                    if i == True:
                        v_list.append(1)
                    else:
                        v_list.append(0)
                return v_list

            v_list = convert(v_list)
            return v_list

        def kill_data_using_v(data_no_move, v):
            v = np.array(v)
            v = torch.tensor(v)
            index = torch.nonzero(v).reshape(-1)
            data_no_move = data_no_move
            index = index.cpu()
            index = index.detach().numpy()
            data_no_move = data_no_move[index, :]
            return data_no_move

        def kill_label_using_v(labels_no_move, v_list):
            labels_no_move = labels_no_move
            item_index = torch.nonzero(torch.tensor(v_list)).reshape(-1)
            final_list = []
            for i in item_index:
                final_list.append(labels_no_move[i])
            return np.array(final_list)

        v_list = convert_v(loss, 0.5)

        data = kill_data_using_v(data_no_moving_scpace, v_list)
        labels = kill_label_using_v(labels_no_moving, v_list)

        predict = est.predict(data_test)
        score_balanced = balanced_accuracy_score(y_true=labels_test, y_pred=predict)
        score = accuracy_score(y_true=labels_test,y_pred=predict)
        print('scpace', score)
        score_max_list.append(score)

        predict_reclassify = est.predict(data_no_moving_scpace)
    return predict_reclassify,score_balanced,score
