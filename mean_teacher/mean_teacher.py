import torch
import torch.nn

def update_mt(model, mt_model, alpha, step):
    """ The brain of the whole thing. Takes two models then sets the
    mean teacher model to use the EMA of the student models """
    alpha = min(1.0 - 1.0 / float(step + 1), alpha)

    for par_t, par in zip(mt_model.parameters(), model.parameters()):
        par_t.data.mul_(alpha).add(1 - alpha, par.data)