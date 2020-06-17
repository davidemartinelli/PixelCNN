import argparse

parser = argparse.ArgumentParser()

#training
parser.add_argument('-e', type=int, default=25, help='Number of training epochs. Default: 25')
parser.add_argument('-b', type=int, default=128, help='Batch size. Default: 128')
parser.add_argument('-lr', type=float, default=1e-3, help='Learning rate. Default: 1e-3')

#model
parser.add_argument('-hl', type=int, default=7, help='Number of hidden layers. Default: 7')
parser.add_argument('-ch', type=int, default=32, help='Number of channels. Default: 32')
parser.add_argument('-k', type=int, default=5, help='Kernel size. Default: 5')

args = parser.parse_args()