import torch
import torch.nn


def update_mt(model, mt_model, alpha, step):
    """ The brain of the whole thing. Takes two models then sets the
    mean teacher model to use the EMA of the student models """
    alpha = min(1 - 1 / (step + 1), alpha)

    for ema_p, param in zip(mt_model.parameters(), model.parameters()):
        ema_p.data.mul_(alpha).add_(1 - alpha, param.data)


class AddGaussianNoise(object):
    def __init__(self, mean=0, std=1):
        super().__init__()
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + 'mean={0}, std={1}'.format(self.mean, self.std)
