import argparse
import os
import os.path
import h5py
import tables
import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


def get_args():
    """ Method to get all commandline arguments for training """
    parser = argparse.ArgumentParser(description="PyTorch Higgs Training")

    parser.add_argument('--epochs', default=20, type=int,
                        metavar='N', help='Total number of epochs to run')

    parser.add_argument('--batch_size', default=1024, type=int,
                        metavar='N', help='training batch size')

    parser.add_argument('--n-labeled', type=int, default=250,
                        help='Number of labeled data to have')

    args = parser.parse_args()

    return args
