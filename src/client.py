import numpy as np


class Client:
    def __init__(self, public_key, private_key, test_x, test_y, p, gamma, verbal=False):
        self.public_key = public_key
        self.private_key = private_key
        self.test_x = test_x
        self.test_y = test_y
        self.verbal = verbal
        self.p = p
        self.gamma = gamma
        self.server = None

    def encrypt(self, num):
        return self.public_key.encrypt(int(num))

    def encrypt_np(self, array):
        encrypted_list = []
        for element in array:
            encrypted_list.append(self.encrypt(element))
        return np.array(encrypted_list)

    def decrypt(self, num):
        return int(self.private_key.decrypt(num))

    def decrypt_np(self, array):
        decrypted_list = []
        for element in array:
            decrypted_list.append(self.decrypt(element))
        return np.array(decrypted_list, dtype='float64')

    def test(self):
        n, _ = self.test_x.shape
        correct_num = 0
        for index in range(0, n):
            test_sample = self.test_x[index]
            encrypted_sample = self.encrypt_np(test_sample * self.gamma)
            test_correct_result = self.test_y[index]
            result = self.server.server_receive('data line', encrypted_sample)
            pred = np.sign(self.decrypt(result))
            print(f'predict {index}')
            if (pred >= 0 and self.test_y[index] > 0) or (pred < 0 and self.test_y[index] <= 0):
                correct_num += 1
                print('correct')
        print(f"correct pred: {correct_num}/{n}={correct_num/n}")

    def client_receive(self, type, data):
        if type == 'kernel':
            dec_masked_kernel = self.decrypt_np(data)
            if self.verbal:
                print(f'CLIENT received: masked degree 1 kernel')
                print(f'content: {dec_masked_kernel}')
                print(f'dtype={dec_masked_kernel.dtype}')
                print()
            raised_masked_kernel = np.power(dec_masked_kernel, self.p)
            if self.verbal:
                print(f'CLIENT sent: encrypted masked degree {self.p} kernel')
                print(f'content: {raised_masked_kernel}')
                print()
            return self.encrypt_np(raised_masked_kernel)
