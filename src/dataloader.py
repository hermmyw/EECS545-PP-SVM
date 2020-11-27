import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm

class DataLoader(object):
    def __init__(self, file_path):
        # import data and remove NaN samples
        self.raw_data = np.genfromtxt(file_path, delimiter=',')
        self.raw_data = self.raw_data[~np.isnan(self.raw_data).any(axis=1)]
        
        # extract features and labels
        X = self.raw_data[:,0:-1]
        y = self.raw_data[:,-1]
        y = self.set_binary(y)
        
        # split and normalize
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, X_test = self.normalize(X_train, X_test)
        self.data = (X_train, X_test, y_train, y_test)
        
    def normalize(self, X, T):
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        T = scaler.transform(T)
        return X, T
    
    def set_binary(self, y, val1=-1, val2=1):
        thr = (np.max(y) + np.min(y))/2
        y[y <= thr] = val1
        y[y > thr] = val2
        return y
        
def get_svm_weights(trainData,testData=None,p=2,c=3,verbose=True):
    '''
    :param trainData: (X_train, Y_train)
    :param testData:  (X_test, Y_test)
    :param p: order of poly (in paper p=2 for wbc dataset)
    :return: support_vecs,intercept, dual_coef, model, perdiction_result(optional)
    '''
    clf = svm.SVC(kernel="poly",degree=p,C=c)
    if verbose:
        scores = cross_val_score(clf, trainData[0], trainData[1], cv=5)
        print("val acc (leave one out among training data):",np.mean(scores))
    clf.fit(trainData[0], trainData[1])
    # support_vecs = clf.support_vectors_
    support_indice = clf.support_
    intercept = clf.intercept_
    dual_coef = clf.dual_coef_

    perdiction_result = None
    if testData !=None:
        perdiction_result = clf.predict(testData[0])
        acc = perdiction_result==testData[1]
        acc = sum(acc)/acc.shape[0]
        if verbose:
            print("test acc:",acc)
    return support_indice, intercept, dual_coef, clf, perdiction_result

if __name__ == '__main__':
    wbc_loader = DataLoader("../assets/breast_cancer_wisconsin.data")
    X_wbc_train, X_wbc_test, y_wbc_train, y_wbc_test = wbc_loader.data
    ddr_loader = DataLoader("../assets/messidor_features.data")
    X_ddr_train, X_ddr_test, y_ddr_train, y_ddr_test = ddr_loader.data

    support_indice,intercept, dual_coef, model,_ =get_svm_weights((X_wbc_train,y_wbc_train),(X_wbc_test,y_wbc_test))

    support_indice_ddr, intercept_ddr, dual_coef_ddr, model_ddr, _ = get_svm_weights((X_ddr_train, y_ddr_train), (X_ddr_test,
                                                                                                y_ddr_test),p=3,c=0.3)
    # Note: ddr dataset is probably not the best dataset for poly SVM, best test accuracy is around 65%