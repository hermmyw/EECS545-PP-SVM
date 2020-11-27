import numpy as np
import dataloader


class Server:
    def __init__(self, public_key, train_x, train_y, p, gamma, verbal=False):
        self.public_key = public_key
        self.train_x = train_x
        self.train_y = train_y
        self.verbal = verbal
        self.p = p
        self.gamma = gamma
        self.train_x, self.train_y, self.alpha, self.b = self.get_parameters(train_x, train_y)
        self.client = None

    def encrypt(self, num):
        return self.public_key.encrypt(int(num))

    def encrypt_np(self, array):
        encrypted_list = []
        for element in array:
            encrypted_list.append(self.encrypt(element))
        return np.array(encrypted_list)

    def get_parameters(self, X, y):
        support_indice, b, alpha, _, _ = dataloader.get_svm_weights((self.train_x, self.train_y), p=self.p, verbose=False)

        return self.train_x[support_indice], self.train_y[support_indice], alpha, b

    def predict(self, vector_t):
        kernel = self.get_kernel(vector_t)
        # print('decrpyted_kernel')
        # print(self.client.decrypt_np(kernel))
        decision_function = np.sum(np.multiply(self.gamma * np.multiply(self.alpha, self.train_y), kernel)) \
                            + self.encrypt(self.gamma**(2*self.p + 1)*self.b)
        print(f"decision_function={self.client.decrypt(decision_function)}")
        # decrypted_t = self.client.decrypt_np(vector_t)
        # decrypted_p1_kernel = []
        # n, d = self.train_x.shape
        # scaled_X = self.gamma * self.train_x
        # for i in range(0, n):
            # decrypted_p1_kernel.append(np.dot(decrypted_t, scaled_X[i]) + self.gamma**2)
        # decrypted_p1_kernel = np.array(decrypted_p1_kernel).astype(int).astype(np.float64)
        # print('normal_p1_kernel')
        # print(decrypted_p1_kernel)
        # decrypted_kernel = np.power(decrypted_p1_kernel, self.p)
        # print('normal_kernel')
        # print(decrypted_kernel)

        return decision_function

    def get_kernel(self, vector_t):
        p1_kernel = self.get_p1kernel_list(vector_t)
        kernel = p1_kernel
        # when the polynomial kernel has a degree more than 1, let client raise the power
        # Corresponding to Equation (17)
        if self.p > 1:
            n, d = self.train_x.shape
            random_mask = (np.random.rand(n)*100+1).astype(int).astype(float)
            # random_mask = np.ones(n) * 2
            # print(np.divide(1, np.power(random_mask, self.p)))
            masked_p1kernel = np.multiply(kernel, random_mask)
            masked_kernel = self.client.client_receive('kernel', masked_p1kernel)

            # print('decrpyted_kernel')
            # print(np.divide(self.client.decrypt_np(masked_kernel), np.power(random_mask, self.p)))


            # kernel = np.divide(masked_kernel, np.power(random_mask, self.p))
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
