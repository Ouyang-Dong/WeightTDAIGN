import numpy as np
import pandas as pd
import math
from data import MDAv2_4_GetData
from method import model


class Experiments(object):
    def __init__(self, mir_dis_data, model_name='WeightTDAIGN', **kwargs):
        super().__init__()
        self.mir_dis_data = mir_dis_data
        self.model = model.WeightTDAIGN_Model(model_name)
        self.parameters = kwargs

    def CV_type(self):
        k_folds = 5
        association_matrix = self.mir_dis_data.type_tensor.sum(2)
        index_matrix = np.array(np.where(association_matrix > 0))
        pair_num = index_matrix.shape[1]
        sample_num_per_fold = int(pair_num / k_folds)
        np.random.seed(0)
        np.random.shuffle(index_matrix.T)
        metrics = 0

        for k in range(k_folds):
            print('{}-fold cross validation'.format(k))
            train_tensor = np.array(self.mir_dis_data.type_tensor, copy=True)
            if k != k_folds - 1:
                test_index = tuple(index_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
            else:
                test_index = tuple(index_matrix[:, k * sample_num_per_fold:])

            train_tensor[test_index] = 0
            train_matrix = train_tensor.sum(2)
            train_matrix[np.where(train_matrix > 0)] = 1


            miRNA_func_similarity_matrix = np.mat(self.mir_dis_data.get_functional_sim(train_matrix))

            nd = train_matrix.shape[1]
            nm = train_matrix.shape[0]

            rd = np.zeros([nd, 1])
            rm = np.zeros([nm, 1])

            for i in range(nd):
                rd[i] = math.pow(np.linalg.norm(train_matrix[:, i]), 2)
            gamad = nd / rd.sum()


            for j in range(nm):
                rm[j] = math.pow(np.linalg.norm(train_matrix[j, :]), 2)
            gamam = nm / rm.sum()


            DGSM = np.zeros([nd, nd])
            for m in range(nd):
                for n in range(nd):
                    DGSM[m, n] = np.exp(
                        -gamad * math.pow(np.linalg.norm(train_matrix[:, m] - train_matrix[:, n]), 2))

            MGSM = np.zeros([nm, nm])
            for r in range(nm):
                for t in range(nm):
                    MGSM[r, t] = np.exp(
                        -gamam * math.pow(np.linalg.norm(train_matrix[r, :] - train_matrix[t, :]), 2))

            miRNA_seq_similarity_matrix = pd.read_csv(
                './HMDD_data/HMDD2.0_processed/MDAv2.0_4/miRNA_seq_sim.csv',
                index_col=0)
            miRNA_seq_sim = np.mat(miRNA_seq_similarity_matrix.values)

            DGSM = np.mat(DGSM)
            MGSM = np.mat(MGSM)

            predict_tensor = self.model()(train_tensor, self.mir_dis_data.dis_sim, DGSM, miRNA_func_similarity_matrix, miRNA_seq_sim,MGSM,
                                          r_1=self.parameters['r_1'], r_2=self.parameters['r_2'], alpha=self.parameters['alpha'],
                                          beta=self.parameters['beta'], lam=self.parameters['lam'],
                                          tol = 1e-3, max_iter = 500)
            TP = 0
            recall = 0
            real_sum = 0

            sample_num_eval = np.array(test_index).shape[1]
            for t in range(sample_num_eval):
                predict_matrix = predict_tensor[test_index[0][t], test_index[1][t]]
                predict_score = np.mat(predict_matrix.flatten())
                real_matrix = self.mir_dis_data.type_tensor[test_index[0][t], test_index[1][t]]
                real_score = np.mat(real_matrix.flatten())
                positive_num = real_score.sum()
                real_sum = real_sum + positive_num

                sort_index = np.array(np.argsort(predict_score))[0]
                predict_score[np.where(predict_score != 0)] = 0
                predict_score[:, sort_index[-1:]] = 1

                tp = predict_score * real_score.T

                TP = TP + tp[0, 0]
                recall = recall + tp[0, 0] / positive_num
            avg_precision = TP / (1 * sample_num_eval)
            mi_avg_recall = TP / real_sum
            ma_avg_recall = recall / sample_num_eval

            metrics = metrics + np.array([avg_precision, mi_avg_recall, ma_avg_recall])

        return metrics / k_folds

    

if __name__ == '__main__':
    root = './HMDD_data'
    mir_dis_data = MDAv2_4_GetData.MDAv2_4_GetData(root)
    experiment = Experiments(mir_dis_data, model_name='WeightTDAIGN', r_1 = 8, r_2 = 20, alpha = 0.02, beta = 0.02, lam = 0.001, tol = 1e-3,
                             max_iter = 500)

    print(experiment.CV_type())




