import csv
import os.path as osp
import numpy as np


class MDAv2_4_GetData(object):
    def __init__(self, root, miRNA_num = 40, dis_num = 20):
        super().__init__()
        self.root = osp.join(root, 'HMDD2.0_processed/MDAv2.0_4')
        self.miRNA_num = miRNA_num
        self.dis_num = dis_num
        self.dis_sim, self.type_tensor = self.__get_data__()

    def __get_data__(self):
        type_name = ['circulation', 'epigenetics', 'genetics','target']
        type_association_matrix = np.zeros((self.miRNA_num, self.dis_num, 4))
        for i in range(4):
            with open(osp.join(self.root, '{}.csv'.format(type_name[i])), 'r') as type_:
                type_mat = csv.reader(type_)
                row = -1

                for line in type_mat:
                    if row >= 0:
                        col = -1
                        for association in line:
                            if col >= 0:
                                type_association_matrix[row, col, i] = eval(association)
                            col = col + 1
                    row = row + 1

        disease_similarity_mat = np.zeros((self.dis_num, self.dis_num))
        with open(osp.join(self.root, 'DSSM2.0_all.csv'), 'r') as dis_sim:
            sim_mat = csv.reader(dis_sim)
            row = -1

            for line in sim_mat:
                if row >= 0:
                    col = -1
                    for sim in line:
                        if col >= 0:
                            disease_similarity_mat[row, col] = eval(sim)
                        col = col + 1
                row = row + 1
        disease_similarity_mat = np.mat(disease_similarity_mat)
        return disease_similarity_mat, type_association_matrix

    def get_functional_sim(self, mir_dis_mat):
        np.seterr(divide='ignore', invalid='ignore')
        mir_fun_sim_matrix = np.zeros((self.miRNA_num, self.miRNA_num))
        dis_semantic_sim = self.dis_sim - np.diag(np.diag(self.dis_sim)) + np.eye(self.dis_num)

        for m1 in range(self.miRNA_num):
            m1_link_num = np.sum(mir_dis_mat[m1]) 
            m1_link_repeat = np.tile(mir_dis_mat[m1], (self.dis_num, 1))
            for m2 in range(m1, self.miRNA_num):
                m2_link_num = np.sum(mir_dis_mat[m2])
                m2_link_repeat = np.tile(mir_dis_mat[m2], (self.dis_num, 1))
                m1_m2_sim_mat = np.multiply(np.multiply(m1_link_repeat, dis_semantic_sim), m2_link_repeat.T)
                m1_max_sum = np.sum(np.max(m1_m2_sim_mat, axis=0))
                m2_max_sum = np.sum(np.max(m1_m2_sim_mat, axis=1))
                mir_fun_sim_matrix[m1, m2] = (m1_max_sum + m2_max_sum) / (m1_link_num + m2_link_num)
                mir_fun_sim_matrix[m2, m1] = (m1_max_sum + m2_max_sum) / (m1_link_num + m2_link_num)

        mir_fun_sim_matrix = np.nan_to_num(mir_fun_sim_matrix)
        row, col = np.diag_indices(mir_fun_sim_matrix.shape[0])
        mir_fun_sim_matrix[row, col] = np.ones(mir_fun_sim_matrix.shape[0])
        return mir_fun_sim_matrix
