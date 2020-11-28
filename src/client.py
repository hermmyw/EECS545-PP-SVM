import numpy as np


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
        encrypted_list = []
        for element in array:
            encrypted_list.append(self.encrypt(element))
        return np.array(encrypted_list)

    '''decrypt an number'''
    def decrypt(self, num):
        return int(self.private_key.decrypt(num))

    '''decrypt an iterable containing encrypted numbers to an np 1-d array with results'''
    def decrypt_np(self, array):
        decrypted_list = []
        for element in array:
            decrypted_list.append(self.decrypt(element))
        return np.array(decrypted_list, dtype='float64')

    '''This is the function to compare a number from server and a number from client'''
    '''in order to simulate the calculation of lambda in the equation after Equation (22)'''
    def magic_comparison(self, num1, num2):
        if self.decrypt(num1) < self.decrypt(num2):
            return 1
        return 0

    def test(self):
        n, _ = self.test_x.shape
        correct_num = 0

        for index in range(0, n):
            print(f'Task {index}/{n}')
            test_sample = self.test_x[index]
            encrypted_sample = self.encrypt_np(test_sample * self.gamma)
            result = self.server.server_receive('data line', encrypted_sample)
            pred = self.decrypt(result)
            print(f'prediction: {pred}')
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
