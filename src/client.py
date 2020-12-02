import numpy as np
import time


class Client:
    def __init__(self, public_key, private_key, test_x, test_y, p, gamma, l, verbal=False):
        self.public_key = public_key  # public key used to encrypt
        self.private_key = private_key  # private key used to decrypt
        self.test_x = test_x  # test set
        self.test_y = test_y  # test set correct labels for performance measurement
        self.verbal = verbal  # verbal mode
        self.p = p  # degree of the polynomial kernel
        self.gamma = gamma  # scaling factor
        self.l = l  # used in decision function
        self.server = None  # server set for communication

    '''encrypt an number'''
    '''if the number is not an integer, it will be converted to integer'''
    def encrypt(self, num):
        return self.public_key.encrypt(int(num))

    '''encrypt an iterable to an np 1-d array with encrypted numbers'''
    def encrypt_np(self, array):
        if self.verbal:
            print(f'CLIENT: encrypting list with size {array.size}')
        start_time = time.time()
        encrypted_list = []
        for element in array:
            encrypted_list.append(self.encrypt(element))
        if self.verbal:
            print(f'CLIENT: finished encrypting matrix with size {array.size}, Time={time.time()-start_time} seconds')
            print()
        return np.array(encrypted_list)

    '''encrypt an iterable to an np 2-d matrix with encrypted numbers'''
    def encrypt_matrix(self, matrix):
        if self.verbal:
            print(f'CLIENT: encrypting matrix with size {matrix.shape}')
        start_time = time.time()
        encrypted_matrix = []
        for vector in matrix:
            encrypted_list = []
            for element in vector:
                encrypted_list.append(self.encrypt(element))
            encrypted_matrix.append(encrypted_list)
        if self.verbal:
            print(f'CLIENT: finished encrypting matrix with size {matrix.shape}, Time={time.time()-start_time} seconds')
            print()
        return np.array(encrypted_matrix)

    '''decrypt an number'''
    def decrypt(self, num):
        return int(self.private_key.decrypt(num))

    '''decrypt an iterable containing encrypted numbers to an np 1-d array with results'''
    def decrypt_np(self, array):
        if self.verbal:
            print(f'CLIENT: encrypting list with size {array.size}')
        start_time = time.time()
        decrypted_list = []
        for element in array:
            decrypted_list.append(self.decrypt(element))
        if self.verbal:
            print(f'CLIENT: finished encrypting matrix with size {array.size}, Time={time.time()-start_time} seconds')
            print()
        return np.array(decrypted_list, dtype='float64')

    '''decrypt an matrix containing encrypted numbers to an np 2-d matrix with results'''
    def decrypt_matrix(self, matrix):
        if self.verbal:
            print(f'CLIENT: decrypting matrix with size {matrix.shape}')
        start_time = time.time()
        n, d = matrix.shape
        decrypted_matrix = np.zeros((n, d), dtype='float64')
        for i in range(0, n):
            for j in range(0, d):
                decrypted_matrix[i, j] = self.decrypt(matrix[i, j])
        if self.verbal:
            print(f'CLIENT: finished encrypting matrix with size {matrix.shape}, Time={time.time()-start_time} seconds')
            print()
        return decrypted_matrix

    '''This is the function to compare a number from server and a number from client'''
    '''in order to simulate the calculation of lambda in the equation after Equation (22)'''
    def magic_comparison(self, num1, num2):
        if self.decrypt(num1) < self.decrypt(num2):
            return 1
        return 0

    '''The comparison made for matrix prediction'''
    def magic_comparison_list(self, np1, np2):
        d_np1 = self.decrypt_np(np1)
        d_np2 = self.decrypt_np(np2)
        n = np1.size
        result = np.zeros(n)
        result[np.greater(d_np2, d_np1)] = 10**self.l
        return result

    def test(self, line_by_line=False):
        n, _ = self.test_x.shape

        if not line_by_line:
            svm_start_time = time.time()
            SVM_matrix_pred = self.server.server_receive('decrypted data matrix', self.test_x)
            svm_end_time = time.time()
            ppsvm_start_time = time.time()
            PPSVM_matrix_pred = self.server.server_receive('data matrix', self.encrypt_matrix(self.test_x * self.gamma))
            PPSVM_matrix_pred = self.decrypt_np(PPSVM_matrix_pred)
            PPSVM_matrix_pred = PPSVM_matrix_pred * 2 - 1  # to make the output either -1 or 1
            ppsvm_end_time = time.time()
            if self.verbal:
                print('CORRECT')
                print(self.test_y)
                print('SVM')
                print(SVM_matrix_pred)
                print('PPSVM')
                print(PPSVM_matrix_pred)
            print(f'SVM TIME: {svm_end_time-svm_start_time} seconds')
            print(f'PPSVM TIME: {ppsvm_end_time-ppsvm_start_time} seconds')

            # Measure the performance
            SVM_accuracy = np.zeros(n)
            PPSVM_accuracy = np.zeros(n)
            similarity = np.zeros(n)
            SVM_accuracy[SVM_matrix_pred == self.test_y] = 1
            PPSVM_accuracy[PPSVM_matrix_pred == self.test_y] = 1
            similarity[PPSVM_matrix_pred == SVM_matrix_pred] = 1
            SVM_accuracy = np.sum(SVM_accuracy)
            PPSVM_accuracy = np.sum(PPSVM_accuracy)
            similarity = np.sum(similarity)
            n_p = np.sum(self.test_y == 1)
            n_n = np.sum(self.test_y == -1) 
            SVM_accuracy_p = np.zeros(n)
            SVM_accuracy_n = np.zeros(n)
            PPSVM_accuracy_p = np.zeros(n)
            PPSVM_accuracy_n = np.zeros(n)
            similarity_p = np.zeros(n)
            similarity_n = np.zeros(n)
            SVM_accuracy_p[(SVM_matrix_pred == 1) & (self.test_y == 1)] = 1
            SVM_accuracy_n[(SVM_matrix_pred == -1) & (self.test_y == -1)] = 1
            PPSVM_accuracy_p[(PPSVM_matrix_pred == 1) & (self.test_y == 1)] = 1
            PPSVM_accuracy_n[(PPSVM_matrix_pred == -1) & (self.test_y == -1)] = 1
            similarity_p[(self.test_y == 1) & (PPSVM_matrix_pred == SVM_matrix_pred)] = 1
            similarity_n[(self.test_y == -1) & (PPSVM_matrix_pred == SVM_matrix_pred)] = 1
            SVM_accuracy_p = np.sum(SVM_accuracy_p)
            SVM_accuracy_n = np.sum(SVM_accuracy_n)
            PPSVM_accuracy_p = np.sum(PPSVM_accuracy_p)
            PPSVM_accuracy_n = np.sum(PPSVM_accuracy_n)
            similarity_p = np.sum(similarity_p)
            similarity_n = np.sum(similarity_n)
            print(f'SVM Accuracy: {SVM_accuracy}/{n}={SVM_accuracy/n}')
            print(f'SVM Accuracy positive class: {SVM_accuracy_p}/{n_p}={SVM_accuracy_p/n_p}')
            print(f'SVM Accuracy negative class: {SVM_accuracy_n}/{n_n}={SVM_accuracy_n/n_n}')
            print(f'PPSVM Accuracy: {PPSVM_accuracy}/{n}={PPSVM_accuracy/n}')
            print(f'PPSVM Accuracy negative class: {PPSVM_accuracy_p}/{n_p}={PPSVM_accuracy_p/n_p}')
            print(f'PPSVM Accuracy positive class: {PPSVM_accuracy_n}/{n_n}={PPSVM_accuracy_n/n_n}')
            print(f'SVM & PPSVM similarity: {similarity}/{n}={similarity/n}')
            print(f'SVM & PPSVM similarity positive class: {similarity_p}/{n_p}={similarity_p/n_p}')
            print(f'SVM & PPSVM similarity negative class: {similarity_n}/{n_n}={similarity_n/n_n}')

        else:
            correct_num = 0
            for index in range(0, n):
                print(f'Task {index}/{n}')
                test_sample = self.test_x[index]
                encrypted_sample = self.encrypt_np(test_sample * self.gamma)
                result = self.server.server_receive('data line', encrypted_sample)
                pred = self.decrypt(result) * 2 - 1  # result of PPSVM is either 0 or 1. change it to -1 or 1
                svm_pred = self.server.server_receive('decrypted data line', test_sample)
                print(f'PPSVM prediction: {pred}')
                print(f'SVM prediction: {svm_pred}')
                print(f'correct: {self.test_y[index]}')
                if (pred > 0 and self.test_y[index] > 0) or (pred <= 0 and self.test_y[index] <= 0):
                    correct_num += 1
                    print('CORRECT')
                print()
            print()
            print(f"correct pred: {correct_num}/{n}={correct_num/n}")

    def client_receive(self, type, data):
        if type == 'kernel':
            # client is called to raise the power of the degree 1 kernel
            dec_masked_kernel = self.decrypt_np(data)
            if self.verbal:
                print(f'CLIENT received: masked degree 1 kernel')
                print(f'content: {dec_masked_kernel}')
                print()
            raised_masked_kernel = np.power(dec_masked_kernel, self.p)
            if self.verbal:
                print(f'CLIENT sent: encrypted masked degree {self.p} kernel')
                print(f'content: {raised_masked_kernel}')
                print()
            return self.encrypt_np(raised_masked_kernel)
        elif type == 'kernel_matrix':
            # client is called to raise the power of the degree 1 matrix kernels
            dec_masked_kernel = self.decrypt_matrix(data)
            if self.verbal:
                print(f'CLIENT received: masked degree 1 kernel')
                print(f'content: {dec_masked_kernel}')
                print()
            raised_masked_kernel = np.power(dec_masked_kernel, self.p)
            if self.verbal:
                print(f'CLIENT sent: encrypted masked degree {self.p} kernel')
                print(f'content: {raised_masked_kernel}')
                print()
            return self.encrypt_matrix(raised_masked_kernel)
        elif type == 'decision function':
            # client is called to perform modulo by 10**l on the masked decision function
            dec_masked_df = self.decrypt(data)
            if self.verbal:
                print(f'CLIENT received: masked decision function')
                print(f'content: {dec_masked_df}')
                print()
            mod_result = dec_masked_df % (10**self.l)  # this is z + r mod 10**l
            if self.verbal:
                print(f'CLIENT sent: masked mod result for decision function')
                print(f'content: {mod_result}')
                print()
            return self.encrypt(mod_result)
        elif type == 'decision function list':
            # client is called to perform modulo by 10**l on the masked decision functions for matrix prediction
            dec_masked_df = self.decrypt_np(data)
            if self.verbal:
                print(f'CLIENT received: masked decision function list')
                print(f'content: {dec_masked_df}')
                print()
            mod_result = np.mod(dec_masked_df, 10**self.l)  # this is z + r mod 10**l
            if self.verbal:
                print(f'CLIENT sent: masked mod results for decision function list')
                print(f'content: {mod_result}')
                print()
            return self.encrypt_np(mod_result)
