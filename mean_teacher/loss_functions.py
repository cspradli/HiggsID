import torch
from torch.nn import functional as F
from torch.autograd import Variable

def softmax_meanse(input, target):
    """ Returns the Mean Squared Error on softmax of both input and output """

    if (input.size() == target.size()):
        input_sm = F.softmax(input, dim=1)
        output_sm = F.softmax(target, dim=1)
        num_classes = input.size()[1]
        return F.mse_loss(input_sm, output_sm, size_average=False) / num_classes
    else:
        print("ERROR softmax_meanse: input and target sizes are not equal")
        return None