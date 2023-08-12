import numpy as np
import tensorly as tl
import math
import scipy.sparse
import warnings
warnings.filterwarnings("ignore")

class WeightTDAIGN_Model(object):

    def __init__(self,name = 'WeightTDAIGN'):
        super().__init__()
        self.name = name

    def WeightTDAIGN(self, X, S_d_1, S_d_2, S_m_1, S_m_2, S_m_3, r_1 = 8, r_2 = 20, alpha = 0.02, beta = 0.02,
                      lam = 0.001, tol = 1e-3, max_iter = 500):

        m = X.shape[0]
        d = X.shape[1]
        t = X.shape[2]


        np.random.seed(0)
        M = np.mat(np.random.rand(m, r_1))
        D = np.mat(np.random.rand(d, r_1))
        T = np.mat(np.random.rand(t, r_1))

        np.random.seed(1)
        U_1 = np.mat(np.random.rand(m, r_2))
        Q_1 = np.mat(np.random.rand(r_2, r_1))
        C_1 = np.mat(np.random.rand(m, r_2))
        F_1 = np.mat(np.random.rand(m, r_2))

        np.random.seed(2)
        U_2 = np.mat(np.random.rand(m, r_2))
        Q_2 = np.mat(np.random.rand(r_2, r_1))
        C_2 = np.mat(np.random.rand(m, r_2))
        F_2 = np.mat(np.random.rand(m, r_2))

        np.random.seed(3)
        U_3 = np.mat(np.random.rand(m, r_2))
        Q_3 = np.mat(np.random.rand(r_2, r_1))
        C_3 = np.mat(np.random.rand(m, r_2))
        F_3 = np.mat(np.random.rand(m, r_2))

        np.random.seed(4)
        V_1 = np.mat(np.random.rand(d, r_2))
        P_1 = np.mat(np.random.rand(r_2, r_1))
        E_1 = np.mat(np.random.rand(d, r_2))
        G_1 = np.mat(np.random.rand(d, r_2))

        np.random.seed(5)
        V_2 = np.mat(np.random.rand(d, r_2))
        P_2 = np.mat(np.random.rand(r_2, r_1))
        E_2 = np.mat(np.random.rand(d, r_2))
        G_2 = np.mat(np.random.rand(d, r_2))


        D_M_1 = np.mat(np.diagflat(S_m_1.sum(1)))
        D_M_2 = np.mat(np.diagflat(S_m_2.sum(1)))
        D_M_3 = np.mat(np.diagflat(S_m_3.sum(1)))
        D_D_1 = np.mat(np.diagflat(S_d_1.sum(1)))
        D_D_2 = np.mat(np.diagflat(S_d_2.sum(1)))


        L_m_1 = D_M_1 - S_m_1
        L_m_2 = D_M_2 - S_m_2
        L_m_3 = D_M_3 - S_m_3
        L_d_1 = D_D_1 - S_d_1
        L_d_2 = D_D_2 - S_d_2



        Y_1 = 0
        Y_2 = 0
        Y_3 = 0
        H_1 = 0
        H_2 = 0
        H_3 = 0
        Z_1 = 0
        Z_2 = 0
        J_1 = 0
        J_2 = 0

        coeff = 1.15
        mu = 1e-6
        gama = 1e-6
        eta = 1e-6
        epsilon = 1e-6

        etamax = 1e6
        mumax = 1e6
        gamamax = 1e6
        epsilonmax = 1e6


        X_1 = np.mat(tl.unfold(X, 0))
        X_2 = np.mat(tl.unfold(X, 1))
        X_3 = np.mat(tl.unfold(X, 2))

        
        W_1 = np.ones(X_1.shape)
        W_2 = np.ones(X_2.shape)
        W_3 = np.ones(X_3.shape)

        temp_dic = {'1': 0.3,'2': 0.3,'3': 0.3,'4': 0.3}
        for v, q in temp_dic.items():
            print('the number of episode:{}'.format(v))
            for i in range(max_iter):
                A = np.mat(tl.tenalg.khatri_rao([D, T]))

                output_X_old = tl.fold(np.array(M * A.T), 0, X.shape)

                new_X_1 = np.multiply(W_1, X_1)
                X_1_G = new_X_1 * A + alpha * U_1 * Q_1 + alpha * U_2 * Q_2 + alpha * U_3 * Q_3

                for j in range(M.shape[0]):
                    W_1_sparse = scipy.sparse.diags(W_1[j, :])
                    M[j, :] = X_1_G[j, :] * np.linalg.pinv(
                        A.T * W_1_sparse * A + alpha * np.eye(r_1) + alpha * np.eye(r_1) + alpha * np.eye(r_1) + lam * np.eye(r_1))


                R = np.mat(tl.tenalg.khatri_rao([M, T]))
                new_X_2 = np.multiply(W_2, X_2)
                X_2_R = new_X_2 * R + beta * V_1 * P_1 + beta * V_2 * P_2
                for m in range(D.shape[0]):
                    W_2_sparse = scipy.sparse.diags(W_2[m, :])
                    D[m, :] = X_2_R[m, :] * np.linalg.pinv(
                        R.T * W_2_sparse * R + beta * np.eye(r_1) + beta * np.eye(r_1) + lam * np.eye(r_1))

                B = np.mat(tl.tenalg.khatri_rao([M, D]))
                new_X_3 = np.multiply(W_3, X_3)
                X_3_B = new_X_3 * B
                for n in range(T.shape[0]):
                    W_3_sparse = scipy.sparse.diags(W_3[n, :])
                    T[n, :] = X_3_B[n, :] * np.linalg.pinv(B.T * W_3_sparse * B + lam * np.eye(r_1))

                temp_U_1 = 2 * alpha * S_m_1.T * C_1 + 2 * alpha * M * Q_1.T + mu * C_1 + Y_1 + H_1 + gama * F_1
                U_1 = temp_U_1 * np.linalg.pinv(2 * alpha * C_1.T * C_1 + 2 * alpha * Q_1 * Q_1.T + mu * np.eye(r_2) + gama * np.eye(r_2))
                
                temp_U_2 = 2 * alpha * S_m_2.T * C_2 + 2 * alpha * M * Q_2.T + mu * C_2 + Y_2 + H_2 + gama * F_2
                U_2 = temp_U_2 * np.linalg.pinv(2 * alpha * C_2.T * C_2 + 2 * alpha * Q_2 * Q_2.T + mu * np.eye(r_2) + gama * np.eye(r_2))

                temp_U_3 = 2 * alpha * S_m_3.T * C_3 + 2 * alpha * M * Q_3.T + mu * C_3 + Y_3 + H_3 + gama * F_3
                U_3 = temp_U_3 * np.linalg.pinv(
                    2 * alpha * C_3.T * C_3 + 2 * alpha * Q_3 * Q_3.T + mu * np.eye(r_2) + gama * np.eye(r_2))

                temp_C_1 = 2 * alpha * S_m_1 * U_1 + mu * U_1 - Y_1
                C_1 = temp_C_1 * np.linalg.pinv(2 * alpha * U_1.T * U_1 + mu * np.eye(r_2))

                temp_C_2 = 2 * alpha * S_m_2 * U_2 + mu * U_2 - Y_2
                C_2 = temp_C_2 * np.linalg.pinv(2 * alpha * U_2.T * U_2 + mu * np.eye(r_2))

                temp_C_3 = 2 * alpha * S_m_3 * U_3 + mu * U_3 - Y_3
                C_3 = temp_C_3 * np.linalg.pinv(2 * alpha * U_3.T * U_3 + mu * np.eye(r_2))

                temp_F_1 = gama * U_1 - H_1
                F_1 = np.linalg.pinv(2 * alpha * L_m_1 + gama * np.eye(L_m_1.shape[0])) * temp_F_1

                temp_F_2 = gama * U_2 - H_2
                F_2 = np.linalg.pinv(2 * alpha * L_m_2 + gama * np.eye(L_m_2.shape[0])) * temp_F_2

                temp_F_3 = gama * U_3 - H_3
                F_3 = np.linalg.pinv(2 * alpha * L_m_3 + gama * np.eye(L_m_3.shape[0])) * temp_F_3


                temp_V_1 = 2 * beta * S_d_1.T * E_1 + 2 * beta * D * P_1.T + eta * E_1 + Z_1 + J_1 + epsilon * G_1
                V_1 = temp_V_1 * np.linalg.pinv(2 * beta * E_1.T * E_1 + 2 * beta * P_1 * P_1.T + eta * np.eye(r_2) + epsilon * np.eye(r_2))

                temp_V_2 = 2 * beta * S_d_2.T * E_2 + 2 * beta * D * P_2.T + eta * E_2 + Z_2 + J_2 + epsilon * G_2
                V_2 = temp_V_2 * np.linalg.pinv(
                    2 * beta * E_2.T * E_2 + 2 * beta * P_2 * P_2.T + eta * np.eye(r_2) + epsilon * np.eye(r_2))

                temp_E_1 = 2 * beta * S_d_1 * V_1 + eta * V_1 - Z_1
                E_1 = temp_E_1 * np.linalg.pinv(2 * beta * V_1.T * V_1 + eta * np.eye(r_2))

                temp_E_2 = 2 * beta * S_d_2 * V_2 + eta * V_2 - Z_2
                E_2 = temp_E_2 * np.linalg.pinv(2 * beta * V_2.T * V_2 + eta * np.eye(r_2))

                temp_G_1 = epsilon * V_1 - J_1
                G_1 = np.linalg.pinv(2 * beta * L_d_1 + epsilon * np.eye(L_d_1.shape[0])) * temp_G_1

                temp_G_2 = epsilon * V_2 - J_2
                G_2 = np.linalg.pinv(2 * beta * L_d_2 + epsilon * np.eye(L_d_2.shape[0])) * temp_G_2

                temp_Q_1 = U_1.T * M
                len_Q_1 = Q_1.shape[0]
                temp_Q_1_list = []
                for q1 in range(len_Q_1):
                    temp_Q_1_list.append(1 / np.linalg.norm(Q_1[q1,:],ord=2))
                temp_diga_Q_1 = scipy.sparse.diags(np.array(temp_Q_1_list))
                Q_1 = np.linalg.pinv(U_1.T * U_1 + 1/2 * temp_diga_Q_1) * temp_Q_1

                temp_Q_2 = U_2.T * M
                len_Q_2 = Q_2.shape[0]
                temp_Q_2_list = []
                for q2 in range(len_Q_2):
                    temp_Q_2_list.append(1 / np.linalg.norm(Q_2[q2, :], ord=2))
                temp_diga_Q_2 = scipy.sparse.diags(np.array(temp_Q_2_list))
                Q_2 = np.linalg.pinv(U_2.T * U_2 + 1/2 * temp_diga_Q_2) * temp_Q_2

                temp_Q_3 = U_3.T * M
                len_Q_3 = Q_3.shape[0]
                temp_Q_3_list = []
                for q3 in range(len_Q_3):
                    temp_Q_3_list.append(1 / np.linalg.norm(Q_3[q3, :], ord=2))
                temp_diga_Q_3 = scipy.sparse.diags(np.array(temp_Q_3_list))
                Q_3 = np.linalg.pinv(U_3.T * U_3 + 1 / 2 * temp_diga_Q_3) * temp_Q_3

                
                temp_P_1 = V_1.T * D
                len_P_1 = P_1.shape[0]
                temp_P_1_list = []
                for p1 in range(len_P_1):
                    temp_P_1_list.append(1 / np.linalg.norm(P_1[p1, :],ord=2))
                temp_diga_P_1 = scipy.sparse.diags(np.array(temp_P_1_list))
                P_1 = np.linalg.pinv(V_1.T * V_1 + 1/2 * temp_diga_P_1) * temp_P_1

                temp_P_2 = V_2.T * D
                len_P_2 = P_2.shape[0]
                temp_P_2_list = []
                for p2 in range(len_P_2):
                    temp_P_2_list.append(1 / np.linalg.norm(P_2[p2, :], ord=2))
                temp_diga_P_2 = scipy.sparse.diags(np.array(temp_P_2_list))
                P_2 = np.linalg.pinv(V_2.T * V_2 + 1 / 2 * temp_diga_P_2) * temp_P_2


                Y_1 = Y_1 + mu * (C_1 - U_1)
                Y_2 = Y_2 + mu * (C_2 - U_2)
                Y_3 = Y_3 + mu * (C_3 - U_3)
                H_1 = H_1 + gama * (F_1 - U_1)
                H_2 = H_2 + gama * (F_2 - U_2)
                H_3 = H_3 + gama * (F_3 - U_3)
                Z_1 = Z_1 + eta * (E_1 - V_1)
                Z_2 = Z_2 + eta * (E_2 - V_2)
                J_1 = J_1 + epsilon * (G_1 - V_1)
                J_2 = J_2 + epsilon * (G_2 - V_2)

                
                mu = min(coeff * mu, mumax)
                eta = min(coeff * eta, etamax)
                gama = min(coeff * gama, gamamax)
                epsilon = min(coeff * epsilon, epsilonmax)

                output_X_new = tl.fold(np.array(T * B.T), 2, X.shape)
                err = np.linalg.norm(output_X_new - output_X_old) / np.linalg.norm(output_X_old)
                if err < tol:
                    break
            
            output_X = tl.fold(np.array(T * B.T), 2, X.shape)

            New_X_1 = np.mat(tl.unfold(output_X, 0))
            New_X_2 = np.mat(tl.unfold(output_X, 1))
            New_X_3 = np.mat(tl.unfold(output_X, 2))

            threshold = math.pow((1-q),2)
            for I in range(New_X_1.shape[0]):
                for S in range(New_X_1.shape[1]):
                    if (math.pow((New_X_1[I, S] - X_1[I, S]), 2) <= threshold ) or (
                            New_X_1[I, S] >= (1 + np.sqrt(threshold))):
                        W_1[I, S] = 1
                    elif (New_X_1[I, S] < 1) and (math.pow((New_X_1[I, S] - X_1[I, S]), 2) > threshold):
                        W_1[I, S] = 1.5

            for O in range(New_X_2.shape[0]):
                for N in range(New_X_2.shape[1]):
                    if (math.pow((New_X_2[O, N] - X_2[O, N]), 2) <= threshold) or (
                            New_X_2[O, N] >= (1 + np.sqrt(threshold))):
                        W_2[O, N] = 1
                    elif (New_X_2[O, N] < 1) and (math.pow((New_X_2[O, N] - X_2[O, N]), 2) > threshold):
                        W_2[O, N] = 1.5

            for K in range(New_X_3.shape[0]):
                for L in range(New_X_3.shape[1]):
                    if (math.pow((New_X_3[K, L] - X_3[K, L]), 2) <= threshold) or (
                            New_X_3[K, L] >= (1 + np.sqrt(threshold))):
                        W_3[K, L] = 1
                    elif (New_X_3[K, L] < 1) and (math.pow((New_X_3[K, L] - X_3[K, L]), 2) > threshold):
                        W_3[K, L] = 1.5

        predict_X = np.array(tl.fold(np.array(M * np.mat(tl.tenalg.khatri_rao([D, T])).T), 0, X.shape))

        return predict_X


    def __call__(self):

        return getattr(self, self.name, None)
