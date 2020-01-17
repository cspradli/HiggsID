import torch
import numpy as np

def gaussian_noise(inputs, mean=0, stdv=0.01):
    """ Method used to Gausssian (normal) noise to inputs """
    