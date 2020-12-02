# %%
from phe import paillier
import numpy as np
import matplotlib.pyplot as plt
import pickle

from dataloader import DataLoader
from client import Client
from server import Server

# %% leave-one-out testing
def accuracy_per_class(y_test, y_pred):
    count = y_test.size
    correct = np.sum(y_pred == y_test)
    acc = [count, correct, correct / count]
    acc_per_cls = []
    for cls in np.unique(y_test):
        cls_count = np.sum(y_test == cls)
        cls_correct = np.sum((y_pred == cls) & (y_test == cls))
        acc_per_cls.append([cls, cls_count, cls_correct, cls_correct / cls_count])
    print("loo test accuracy:", acc)
    print("loo test accuracy per class:", acc_per_cls)
    return acc, acc_per_cls

def PPSVM_loo(file_path="../assets/breast_cancer_wisconsin.data", 
              verbose=False, p=2, c=3, s=4, l=9):
    gamma = 10**s
    line = False
    t = 0.1

    wbc_loader = DataLoader(file_path, t)
    loo_list = wbc_loader.data_loo
    n = len(loo_list)
    y_test = np.zeros(n)
    y_pred_SVM = np.zeros(n)
    y_pred_PPSVM = np.zeros(n)
    n_sv = np.zeros(n)
    time_SVM = np.zeros(n)
    time_PPSVM = np.zeros(n)
    
    for i, train_test in enumerate(loo_list):
        print(f"data point {i+1}/{n}")
        y_test[i] = train_test[3]

        # generate encryption/decryption keys
        public_key, private_key = paillier.generate_paillier_keypair()
        # initializing client and server with information that each entity is supposed to know
        client = Client(public_key, private_key, train_test[1], train_test[3], p, gamma, l, verbose)
        server = Server(public_key, train_test[0], train_test[2], p, c, gamma, l, verbose)
        client.server = server
        server.client = client
        # start predicting/testing
        client.test(line)
        
        n_sv[i] = server.train_y.size
        y_pred_SVM[i] = client.y_SVM
        y_pred_PPSVM[i] = client.y_PPSVM
        time_SVM[i] = client.time_SVM
        time_PPSVM[i] = client.time_PPSVM
    
    print("loo average number of support vectors:", np.mean(n_sv))
    acc_SVM, acc_per_cls_SVM = accuracy_per_class(y_test, y_pred_SVM)
    acc_PPSVM, acc_per_cls_PPSVM = accuracy_per_class(y_test, y_pred_PPSVM)
    similarity = np.sum(y_pred_SVM == y_pred_PPSVM) / n
    print("SVM total time:", np.sum(time_SVM))
    print("PPSVM total time:", np.sum(time_PPSVM))
    print("similarity:", similarity)
    result_SVM = [y_pred_SVM, acc_SVM, acc_per_cls_SVM, time_SVM]
    result_PPSVM = [y_pred_PPSVM, acc_PPSVM, acc_per_cls_PPSVM, time_PPSVM]

    return n_sv, result_SVM, result_PPSVM, similarity

# %% run on WBC dataset
s_list = list(range(7))
result_wbc_s = []
for s in s_list:
    result_wbc_s.append(PPSVM_loo(file_path="../assets/breast_cancer_wisconsin.data", 
                                  verbose=False, p=2, c=3, s=s, l=9))

with open("../output/result_wbc_s.pkl", 'wb') as f:
    pickle.dump(result_wbc_s, f)

# %% make figures
acc_SVM = np.array([[result_wbc_s[s][1][1][2] for s in s_list], \
                    [result_wbc_s[s][1][2][0][3] for s in s_list],
                    [result_wbc_s[s][1][2][1][3] for s in s_list]])
acc_PPSVM = np.array([[result_wbc_s[s][2][1][2] for s in s_list], \
                      [result_wbc_s[s][2][2][0][3] for s in s_list],
                      [result_wbc_s[s][2][2][1][3] for s in s_list]])
time_PPSVM = np.array([np.sum(result_wbc_s[s][2][3]) for s in s_list])
scale = 10 ** np.array(s_list)

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=600)
axs[0].semilogx(scale, acc_SVM[0, :], label='SVM total', linestyle='--', color='C0')
axs[0].semilogx(scale, acc_SVM[1, :], label='SVM negative', linestyle='--', color='C1')
axs[0].semilogx(scale, acc_SVM[2, :], label='SVM positive', linestyle='--', color='C2')
axs[0].semilogx(scale, acc_PPSVM[0, :], label='PPSVM total', linestyle='-', color='C0')
axs[0].semilogx(scale, acc_PPSVM[1, :], label='PPSVM negative', linestyle='-', color='C1')
axs[0].semilogx(scale, acc_PPSVM[2, :], label='PPSVM positive', linestyle='-', color='C2')
axs[0].legend()
axs[0].set_xlabel('scale')
axs[0].set_ylabel('accuracy')
axs[1].semilogx(scale, time_PPSVM)
axs[1].set_ylim([0, 2000])
axs[1].set_xlabel('scale')
axs[1].set_ylabel('time (s)')
plt.tight_layout()
fig.savefig('../output/result_wbc_s.pdf')

# %%
