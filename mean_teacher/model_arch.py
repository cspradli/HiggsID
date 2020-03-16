import math
#import torch
import torch.nn as nn


def seq_model_5(input, layer_arr, output, dimen, ema=False):
    """ Create a sequential array based off of the inputs, layer array must be 6 layers """
    model = nn.Sequential(nn.Linear(input, layer_arr[0]),
                        nn.ReLU(),
                        nn.Linear(layer_arr[0], layer_arr[1]),
                        nn.ReLU(),
                        nn.Linear(layer_arr[1], layer_arr[2]),
                        nn.ReLU(),
                        nn.Linear(layer_arr[2], layer_arr[3]),
                        nn.ReLU(),
                        nn.Linear(layer_arr[3], layer_arr[4]),
                        nn.ReLU(),
                        nn.Linear(layer_arr[4], layer_arr[5]),
                        nn.ReLU(),
                        nn.Linear(layer_arr[5], output),
                        nn.LogSoftmax(dim=dimen)
                        )

    if ema:
        for param in model.parameters():
            param.detach_()
            
    return model


def seq_model_6(input, layer_arr, output, dimen, ema=False):
    """ Create a sequential array based off of the inputs, layer array must be 6 layers """
    model = nn.Sequential(nn.Linear(input, layer_arr[0]),
                        nn.ReLU(),
                        nn.Linear(layer_arr[0], layer_arr[1]),
                        nn.ReLU(),
                        nn.Linear(layer_arr[1], layer_arr[2]),
                        nn.ReLU(),
                        nn.Linear(layer_arr[2], layer_arr[3]),
                        nn.ReLU(),
                        nn.Linear(layer_arr[3], layer_arr[4]),
                        nn.ReLU(),
                        nn.Linear(layer_arr[4], layer_arr[5]),
                        nn.ReLU(),
                        nn.Linear(layer_arr[5], layer_arr[6]),
                        nn.ReLU(),
                        nn.Linear(layer_arr[6], output),
                        nn.LogSoftmax(dim=dimen)
                        )

    if ema:
        for param in model.parameters():
            param.detach_()
            
    return model

def creat_seq_model(ema=False):
    """Create a basic neural network with predetermined sizes """
    model = nn.Sequential(nn.Linear(27, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 2),
                      nn.LogSoftmax(dim=1))

    if ema:
        for param in model.parameters():
            param.detach_()
            
    return model
    


