import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm

class DataLoader(object):
    def __init__(self, file_path, test_size, need_normalize=True):
        # import data and remove NaN samples
        self.raw_data = np.genfromtxt(file_path, delimiter=',')
        self.raw_data = self.raw_data[~np.isnan(self.raw_data).any(axis=1)]
        print("Data size: ", self.raw_data.shape)

        # extract features and labels
        X = self.raw_data[:,0:-1]
        y = self.raw_data[:,-1]
        y = self.set_binary(y)
        
        # split and normalize
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        if need_normalize:
            X_train, X_test = self.normalize(X_train, X_test)
        self.data = (X_train, X_test, y_train, y_test)

        # leave-one-out
        self.data_loo = []
        for i in range(y.size):
            indices_loo = list(range(i)) + list(range(i+1, y.size))
            X_train_loo = X[indices_loo]
            X_test_loo = X[[i]]
            y_train_loo = y[indices_loo]
            y_test_loo = y[[i]]
            if need_normalize:
                X_train_loo, X_test_loo = self.normalize(X_train_loo, X_test_loo)
            self.data_loo.append((X_train_loo, X_test_loo, y_train_loo, y_test_loo))

        
    def normalize(self, X, T=None):
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        if T is not None:
            T = scaler.transform(T)
            return X, T
        return X
    
    def set_binary(self, y, val1=-1, val2=1):
        thr = (np.max(y) + np.min(y))/2
        y[y <= thr] = val1
        y[y > thr] = val2
        return y
        
def get_svm_weights(trainData,testData=None,p=2,c=1,verbose=True):
    '''
    :param trainData: (X_train, Y_train)
    :param testData:  (X_test, Y_test)
    :param p: order of poly (in paper p=2 for wbc dataset)
    :return: support_vecs,intercept, dual_coef, model, perdiction_result(optional)
    '''
    # def poly_kernel(X1, X2):
    #     return (X1 @ X2.T + 1) ** 2
    # clf = svm.SVC(kernel=poly_kernel, C=c)
    clf = svm.SVC(kernel="poly", degree=p, gamma=1.0, coef0=1.0, C=c)
    if verbose:
        scores = cross_val_score(clf, trainData[0], trainData[1], cv=5)
        print("val acc (leave one out among training data):",np.mean(scores))
    clf.fit(trainData[0], trainData[1])
    # support_vecs = clf.support_vectors_
    support_indice = clf.support_
    intercept = clf.intercept_
    dual_coef = clf.dual_coef_

    prediction_result = None
    if testData !=None:
        prediction_result = clf.predict(testData[0])
        decision_functions = clf.decision_function(testData[0])
        acc = prediction_result==testData[1]
        acc = sum(acc)/acc.shape[0]
        acc_per_cls = []
        for cls in np.unique(testData[1]):
            counts = np.sum(testData[1] == cls)
            counts_correct = np.sum((prediction_result == cls) & (testData[1] == cls))
            acc_per_cls.append([cls, counts, counts_correct, counts_correct / counts])
        if verbose:
            print("test acc:",acc)
            print("test acc per class:",acc_per_cls)
    return support_indice, intercept, dual_coef, clf, prediction_result

def test_loo(loo_list, p=2, c=3, verbose=False):
    y_test = np.zeros(len(loo_list))
    y_pred = np.zeros(len(loo_list))
    n_sv = np.zeros(len(loo_list))
    for i, train_test in enumerate(loo_list):
        support_indice, _, _, _, prediction_result = \
            get_svm_weights((train_test[0], train_test[2]), (train_test[1], train_test[3]), p=p, c=c, verbose=verbose)
        y_test[i] = train_test[3]
        y_pred[i] = prediction_result
        n_sv[i] = support_indice.size
    print("loo average number of support vectors:", np.mean(n_sv))
    acc = (y_pred == y_test)
    acc = sum(acc) / acc.size
    acc_per_cls = []
    for cls in np.unique(y_test):
        counts = np.sum(y_test == cls)
        counts_correct = np.sum((y_pred == cls) & (y_test == cls))
        acc_per_cls.append([cls, counts, counts_correct, counts_correct / counts])
    print("loo test acc:", acc)
    print("loo test acc per class:", acc_per_cls)
    return y_pred, n_sv, acc, acc_per_cls

if __name__ == '__main__':
    wbc_loader = DataLoader("../assets/breast_cancer_wisconsin.data", 0.3)
    X_wbc_train, X_wbc_test, y_wbc_train, y_wbc_test = wbc_loader.data
    ckd_loader = DataLoader("../assets/chronic_kidney_disease2.data", 1/228)
    X_ckd_train, X_ckd_test, y_ckd_train, y_ckd_test = ckd_loader.data

    support_indice,intercept, dual_coef, model,_ =get_svm_weights((X_wbc_train,y_wbc_train),(X_wbc_test,y_wbc_test), p=2, c=1)

    #support_indice_ddr, intercept_ddr, dual_coef_ddr, model_ddr, _ = get_svm_weights((X_ddr_train, y_ddr_train), (X_ddr_test,
                                                                                                #y_ddr_test),p=3,c=0.3)
    # Note: ddr dataset is probably not the best dataset for poly SVM, best test accuracy is around 65%
    support_indice_ckd, intercept_ckd, dual_coef_ckd, model_ckd, _ = get_svm_weights((X_ckd_train, y_ckd_train), (X_ckd_test,
                                                                                                y_ckd_test),p=3,c=0.3)
    

    test_loo(wbc_loader.data_loo, p=2, c=1, verbose=False) 