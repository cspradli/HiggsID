import math
import torch
import torch.nn as nn



def creat_seq_model(input_arr1, input_arr2):
    """Create a basic neural network with predetermined sizes """
    model = nn.Sequential(nn.Linear(23, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 2),
                      nn.LogSoftmax(dim=1))

    return model
    


