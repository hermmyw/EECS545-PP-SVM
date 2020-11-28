import numpy as np
import math
import dataloader
import random  # use of random instead of np.random as there's a int32 limit to np


class Server:
    def __init__(self, public_key, train_x, train_y, p, gamma, l, verbal=False):
        self.public_key = public_key  # public key used to encrypt
        self.train_x = train_x  # training set
        self.train_y = train_y  # training set labels
        self.verbal = verbal  # verbal mode
        self.p = p  # degree of the polynomial kernel
        self.gamma = gamma  # scaling factor
        self.l = l  # used in decision function
        self.train_x, self.train_y, self.alpha, self.b = self.get_parameters(train_x, train_y)  # model parameters
        self.client = None  # client set for communication

    '''encrypt an number'''
    '''if the number is not an integer, it will be converted to integer'''
    def encrypt(self, num):
        return self.public_key.encrypt(int(num))

    '''encrypt an iterable to an np 1-d array with encrypted numbers'''
    def encrypt_np(self, array):
        encrypted_list = []
        for element in array:
            encrypted_list.append(self.encrypt(element))
        return np.array(encrypted_list)

    '''pre-train the svm parameters. The procedure is the same for normal SVM, thus sklearn is used for now'''
    def get_parameters(self, X, y):
        support_indice, b, alpha, _, _ = dataloader.get_svm_weights((self.train_x, self.train_y), p=self.p, verbose=False)

        return self.train_x[support_indice], self.train_y[support_indice], alpha, b

    '''Takes an encrypted vector t as parameter, output class lable (0 or 1) for the vector'''
    def predict(self, vector_t):
        kernel = self.get_kernel(vector_t)
        decision_function = np.sum(np.multiply(self.gamma * np.multiply(self.alpha, self.train_y), kernel)) \
                            + self.encrypt(self.gamma**(2*self.p + 1)*self.b)
        print(f'decision function: {self.client.decrypt(decision_function)}')
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
        prediction = unmasked_result / (10**self.l)

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
        scaled_X = self.gamma * self.train_x
        p1kernel_list = []
        n, d = self.train_x.shape
        for i in range(0, n):
            p1kernel_list.append(np.dot(vector_t, scaled_X[i]) + self.encrypt(self.gamma**2))
        return np.array(p1kernel_list)

    def server_receive(self, type, data):
        if self.verbal:
            print(f'SERVER received: {type}')
            print(f'content: {data}')
            print()
        if type == 'data line':
            return self.predict(data)
