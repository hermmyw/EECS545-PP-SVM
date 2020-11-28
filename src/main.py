from phe import paillier
import argparse

from dataloader import DataLoader
from client import Client
from server import Server

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', nargs=1, default=["../assets/breast_cancer_wisconsin.data"],
                    type=str, help='data file location, default=\"../assets/breast_cancer_wisconsin.data\"')
parser.add_argument("-v", "--verbose", help="run with debug output", action="store_true")
parser.add_argument("-p", "--degree", help="the degree of the polynomial kernel", default=1, type=int)
parser.add_argument("-s", "--scale", help="the scaling factor gamma, as numbers will be scaled to 10**s",
                    default=1, type=int)
parser.add_argument("-t", "--test", help="proportion of test set, range from (0, 1)",
                    default=0.2, type=float)
parser.add_argument("-l", "--digits", help="choice of l in Section 3.3.4. Default value should be large enough",
                    default=10, type=int)
args=parser.parse_args()

file_path = args.file[0]
verbose = args.verbose
p = args.degree
gamma = 10**args.scale
l = args.digits
t = args.test

# load data
wbc_loader = DataLoader(file_path, t)
train_X, test_X, train_y, test_y = wbc_loader.data

# generate encryption/decryption keys
public_key, private_key = paillier.generate_paillier_keypair()

# initializing client and server with information that each entity is supposed to know
client = Client(public_key, private_key, test_X, test_y, p, gamma, l, verbose)
server = Server(public_key, train_X, train_y, p, gamma, l, verbose)
client.server = server
server.client = client

client.test()