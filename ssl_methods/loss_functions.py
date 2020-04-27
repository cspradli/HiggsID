
from torch.nn import functional as F
from torch.autograd import Variable

def softmax_meanse(input, target):
    """ Returns the Mean Squared Error on softmax of both input and output """

    assert input.size() == target.size()
    input_sm = F.softmax(input, dim=1)
    output_sm = F.softmax(target, dim=1)
    num_classes = input.size()[1]
    return F.mse_loss(input_sm, output_sm, size_average=False) / num_classes
    