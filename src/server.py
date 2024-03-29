import numpy as np
import math
import time
import dataloader
import random  # use of random instead of np.random as there's a int32 limit to np


class Server:
    def __init__(self, public_key, train_x, train_y, p, c, gamma, l, verbal=False):
        self.public_key = public_key  # public key used to encrypt
        self.train_x = train_x  # training set
        self.train_y = train_y  # training set labels
        self.verbal = verbal  # verbal mode
        self.p = p  # degree of the polynomial kernel
        self.c = c  # regularization parameter
        self.gamma = gamma  # scaling factor
        self.l = l  # used in decision function
        self.mean, self.std = self.get_normalize_factor(train_x)  # get mean and std dev of train_x for normalization
        self.train_x = np.divide(np.subtract(self.train_x, self.mean), self.std)  # normalize train_x
        self.train_x, self.train_y, self.alpha, self.b = self.get_parameters(self.train_x, train_y)  # model parameters
        self.client = None  # client set for communication

    '''encrypt an number'''
    '''if the number is not an integer, it will be converted to integer'''
    def encrypt(self, num):
        return self.public_key.encrypt(int(num))

    '''encrypt an iterable to an np 1-d array with encrypted numbers'''
    def encrypt_np(self, array):
        if self.verbal:
            print(f'SERVER: encrypting list with size {array.size}')
        start_time = time.time()
        encrypted_list = []
        for element in array:
            encrypted_list.append(self.encrypt(element))
        if self.verbal:
            print(f'SERVER: finished encrypting matrix with size {array.size}, Time={time.time()-start_time} seconds')
            print()
        return np.array(encrypted_list)

    '''encrypt an iterable to an np 2-d matrix with encrypted numbers'''
    def encrypt_matrix(self, matrix):
        if self.verbal:
            print(f'SERVER: encrypting matrix with size {matrix.shape}')
        start_time = time.time()
        encrypted_matrix = []
        for vector in matrix:
            encrypted_list = []
            for element in vector:
                encrypted_list.append(self.encrypt(element))
            encrypted_matrix.append(encrypted_list)
        if self.verbal:
            print(f'SERVER: finished encrypting matrix with size {matrix.shape}, Time={time.time()-start_time} seconds')
            print()
        return np.array(encrypted_matrix)

    def get_normalize_factor(self, train_x):
        std = np.std(train_x, axis=0, ddof=1)
        mean = np.mean(train_x, axis=0)
        return mean, std

    '''pre-train the svm parameters. The procedure is the same for normal SVM, thus sklearn is used for now'''
    def get_parameters(self, X, y):
        support_indice, b, alpha, _, _ = dataloader.get_svm_weights((X, y), p=self.p, c=self.c, verbose=False)
        if self.verbal:
            print(f'SERVER: number of support vectors {support_indice.size}')
            print(support_indice)
        return X[support_indice], y[support_indice], alpha, b

    '''normalize the data client sent'''
    def normalize(self, t):
        substr = self.encrypt_np(np.multiply(np.divide(self.gamma, self.std), self.mean))
        normalized_data = np.subtract(np.divide(t, self.std), substr)
        return normalized_data

    '''Takes an encrypted vector t as parameter, output class label (0 or 1) for the vector'''
    def predict(self, vector_t):
        vector_t = self.normalize(vector_t)
        kernel = self.get_kernel(vector_t)
        decision_function = np.sum(np.multiply(self.gamma * self.alpha, kernel)) \
                            + self.encrypt(self.gamma**(2*self.p + 1)*self.b)
        print(f'PPSVM decision function: {self.client.decrypt(decision_function)}')
        '''as the result of decision function usually has too many digits, and it is only the sign of it that matters'''
        '''here we conduct a digit deduction, which does not influence the sign of the decision function'''
        min_digits = int(math.log10(abs(self.gamma**(2*self.p + 1)*self.b)))
        decision_function /= (10**min_digits)
        z = decision_function + self.encrypt(10**self.l)
        r = random.randint(1, 10**self.l-1)  # use random instead of np.random as np has a type limit on int32
        masked_z = z + self.encrypt(r)

        masked_result = self.client.client_receive('decision function', masked_z)
        # Equation 21/22
        unmasked_result = masked_result - self.encrypt(r % (10**self.l))
        if self.client.magic_comparison(masked_result, self.encrypt(r % (10**self.l))) == 1:
            unmasked_result += self.encrypt(10**self.l)
        unmasked_result = z - unmasked_result
        prediction = unmasked_result / (10**self.l)  # the prediction in form of 0 or 1

        return prediction

    '''calculate the kernel for the encrypted vector t'''
    def get_kernel(self, vector_t):
        p1_kernel = self.get_p1kernel_list(vector_t)
        kernel = p1_kernel
        # when the polynomial kernel has a degree more than 1, let client raise the power
        # Corresponding to Equation (17)
        if self.p > 1:
            n, d = self.train_x.shape
            random_mask = (np.random.rand(n)*100+1).astype(int).astype(float)
            masked_p1kernel = np.multiply(kernel, random_mask)
            masked_kernel = self.client.client_receive('kernel', masked_p1kernel)
            kernel = np.multiply(masked_kernel, np.divide(1, np.power(random_mask, self.p)))
        return kernel

    '''corresponding to Equation (15)(16)'''
    def get_p1kernel_list(self, vector_t):
        scaled_X = (self.gamma * self.train_x).astype(int)
        p1kernel_list = []
        n, d = self.train_x.shape
        for i in range(0, n):
            p1kernel_list.append(np.dot(vector_t, scaled_X[i]) + self.encrypt(self.gamma**2))
        return np.array(p1kernel_list)

    def predict_matrix(self, matrix_t):
        matrix_t = self.normalize(matrix_t)
        n, d = matrix_t.shape
        '''calculate the kernels for the encrypted vector t'''
        kernels = self.get_kernel_matrix(matrix_t)
        print('calculating decision functions')
        start_time = time.time()
        decision_functions = np.multiply((self.gamma * self.alpha).astype(int), kernels)
        decision_functions = np.sum(decision_functions, axis=1) + self.encrypt(self.gamma ** (2 * self.p + 1) * self.b)
        min_digits = int(math.log10(abs(self.gamma ** (2 * self.p + 1) * self.b)))
        decision_functions = np.divide(decision_functions, 10 ** min_digits)
        print(f'finished calculating decision functions, Time={time.time()-start_time}')
        print()

        print('calculating signs of decision functions')
        start_time = time.time()
        # with the decision function, trying to calculate the signs
        z = np.add(decision_functions, self.encrypt(10 ** self.l))
        r = np.random.randint(low=1, high=10 ** self.l - 1, size=n)
        mod_r = np.mod(r, 10 ** self.l)
        masked_z = np.add(z, r)

        masked_result = self.client.client_receive('decision function list', masked_z)
        # Equation 21/22
        unmasked_result = np.subtract(masked_z, mod_r)
        underflow_flag = self.client.magic_comparison_list(masked_result, self.encrypt_np(mod_r))
        unmasked_result = np.add(unmasked_result, underflow_flag)
        unmasked_result = np.subtract(z, unmasked_result)
        predictions = np.add(np.divide(unmasked_result, 10 ** self.l), self.encrypt(1))  # the predictions in form of list of 0 or 1
        print(f'finished calculating signs of decision functions, Time={time.time()-start_time}')
        print()

        return predictions

    def get_kernel_matrix(self, matrix_t):
        n, d = self.train_x.shape
        test_n, _ = matrix_t.shape
        print('calculating degree 1 kernel')
        start_time = time.time()
        p1_kernel = np.add(np.dot(matrix_t, np.transpose(self.gamma * self.train_x).astype(int)), self.gamma**2)
        end_time = time.time()
        print(f'finished calculating degree 1 kernel, Time={end_time-start_time} seconds')
        print()
        kernel = p1_kernel
        # when the polynomial kernel has a degree more than 1, let client raise the power
        # Corresponding to Equation (17)
        if self.p > 1:
            print(f'calculating degree {self.p} kernel')
            start_time = time.time()
            random_mask = np.random.randint(1, 100, size=(test_n, n))
            masked_p1kernel = np.multiply(kernel, random_mask)
            masked_kernel = self.client.client_receive('kernel_matrix', masked_p1kernel)
            kernel = np.divide(masked_kernel, np.power(random_mask, self.p))
            end_time = time.time()
            print(f'finished calculating degree {self.p} kernel, Time={end_time-start_time} seconds')
            print()
        return kernel

    '''predict using normal SVM'''
    '''used for comparison, not part of the PPSVM'''
    def decrypted_predict(self, vector_t):
        vector_t = np.divide(np.subtract(vector_t, self.mean), self.std)
        print(vector_t)

        kernel = np.power(np.add(np.dot(self.train_x, vector_t), 1), self.p)
        decision_function = np.multiply(self.alpha, kernel)
        decision_function = (np.sum(decision_function) + self.b)[0]
        print(f'SVM decision function: {decision_function}')
        if decision_function < 0:
            return -1
        return 1

    '''predict a matrix using normal SVM'''
    '''used for comparison, not part of the PPSVM'''
    def decrypted_predict_matrix(self, matrix_t):
        matrix_t = np.divide(np.subtract(matrix_t, self.mean), self.std)

        p1_kernel = np.add(np.dot(self.train_x, np.transpose(matrix_t)), 1)  # K(x, t) = (xt + 1)
        kernel = np.power(p1_kernel, self.p)
        decision_functions = np.multiply(np.transpose(self.alpha), kernel)
        decision_functions = (np.sum(decision_functions, axis=0) + self.b)
        results = np.sign(decision_functions)
        return results

    def server_receive(self, type, data):
        if self.verbal and type != 'decrypted data line' and type != 'decrypted data matrix':
            print(f'SERVER received: {type}')
            print(f'content: {data}')
            print()
        if type == 'data line':
            return self.predict(data)
        if type == 'data matrix':
            return self.predict_matrix(data)
        elif type == 'decrypted data line':
            return self.decrypted_predict(data)
        elif type == 'decrypted data matrix':
            return self.decrypted_predict_matrix(data)
