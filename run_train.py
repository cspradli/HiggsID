import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

import utils
import time
import numpy as np
import os.path

from mean_teacher.mean_teacher import AddGaussianNoise
from mean_teacher import dataset
from mean_teacher import loss_functions
from mean_teacher import mean_teacher
from mean_teacher import model_arch
import args_util
import warnings

### Get rid of warnings from Torch ###
### Make sure device is CPU if no CUDA enables device ###
warnings.simplefilter("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
### Get args from command-line ###
args = args_util.get_args()

#Debug printing#
print(args.epochs)
print(args.batch_size)
print(args.percent_unlabeled)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def linear_rampup(current, rampup_length=args.epochs):i
    """ linear rampup from MixMatch paper, rampup used in SSL training """
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    """ Get the semi-supervised loss as described in MixMatch paper """
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, 75 * linear_rampup(epoch)

def train(train_loader, unlabeled_loader, model, mt_model, optimizer, epoch, ema_const=0.95):
    """ Main training method, incorporates all SSL methods from MixMatch paper, in addition
    to adding in more MeanTeacher methods (EMA modeling) """
    global global_step
    ### Meters used for monitoring loss and accuracy during trainign ###
    losses1 = utils.AverageMeter()
    lossesmt = utils.AverageMeter()
    losses1_vt = utils.AverageMeter()
    losses2_vt = utils.AverageMeter()
    ws = utils.AverageMeter()
    losses1_vtmt = utils.AverageMeter()
    losses2_vtmt = utils.AverageMeter()
    wsmt = utils.AverageMeter()
    ## Choose between loss criterion ##
    criterion_ce = nn.CrossEntropyLoss(size_average=False)
    criterion_train = SemiLoss()
    criterion = nn.NLLLoss(size_average=False)
    criterion_mse = nn.MSELoss(size_average=False)
    criterion_kl = nn.KLDivLoss(size_average=False)
    criterion_l1 = nn.L1Loss(size_average=False)

    ##Running loss for output ##
    run_loss = 0
    run_loss_mt = 0
    run_loss_vt = 0
    run_loss_mt_vt = 0

    consistency_criterion = loss_functions.softmax_meanse


    # calculate the amount of unlabelled #
    #num_unlabelled = len(dat_loader) - (args.percent_labeled * len(dat_loader))

    ## Set both models to training mode ##
    model.train()
    mt_model.train()

    enum_Xloader = iter(train_loader)
    enum_Uloader = iter(unlabeled_loader)
    #print(enum_Xloader.shape())
    for i in range(args.batch_size):

        ## Try to get all the next datasets in range of the batch_size
        try:
            images, labels = next(enum_Xloader)
        except StopIteration:
            #print("ERROR: Exception 1")
            enum_Xloader = iter(train_loader)
            images, labels = next(enum_Xloader)


        try:
            (uX, _) = enum_Uloader.next()
        except:
            #print("ERROR: exception 2")
            enum_Uloader = iter(unlabeled_loader)
            (uX, _) = enum_Uloader.next()

        global_step += 1

        images = images.view(images.shape[0], -1)
        #batch_size = images.size(0)
        sl = images.shape
        minibatch_size = len(labels)
        
        ### Convert (or make sure) that all Torch variables used are Torch-friendly variables ###
        input_var = Variable(images)
        mt_input = Variable(images)
        target_var = Variable(labels)
        batch_size = images.size(0)

        ## Compute guessed labels for unlabelled##
        ## From methods outlines in MixMatch ##
        ## Creates a one-hot label (i.e. high-confidence prediction) from unlabeled data ##
        with torch.no_grad():
            outputs_u = mt_model(mt_input)
            p = (torch.softmax(outputs_u, dim=1))
            pt = p**(1/0.5)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()
        

        ## Mixup as discribed in paper ##
        ## Combining newly 'labeled' data with actual labeled data ##
        ## We can treat this as supervised learning training now ##
        all_inputs = torch.cat([images, uX], dim=0)
        all_targets = torch.cat([labels, targets_u], dim=0)

        l = np.random.beta(0.75, 0.75)
        l = max(1, 1-l)
        index = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[index]
        target_a, target_b = all_targets, all_targets[index]

        mixed_input = 1 * input_a + (1 - l) * input_b
        mixed_target = 1 * target_a + (1 - l) * target_b

        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        ## Lx, Lu, w are all loss terms as defined in MixMatch ##
        ## In PyTorch, we used the built-in criterion on our data ##
        # Lx = Loss for supervised aspect (labeled data)#
        # Lu = Loss for unlabeled aspect #
        Lx, Lu, w = criterion_train(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+i/args.val_iteration)
        loss = Lx + w * Lu


        ### Update losses based on the Torch loss function and sum of Lx, w * Lu ##
        losses1.update(loss.item(), images.size(0))
        losses1_vt.update(Lx.item(), images.size(0))
        losses2_vt.update(Lu.item(), images.size(0))
        ws.update(w, images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        ## Update the EMA model based on MT methods ##
        ## Takes an EMA step to update the weights of this specific model ##
        mean_teacher.update_mt(model, mt_model, ema_const, global_step)
        
        ## Sum up loss from total batch ##
        end = time.time()
        run_loss += loss.item()

    else:
        ## Once we've reached end of epoch, publish losses ##
        plotter1.plot('Loss', 'model', 'Model Loss', epoch, losses1.avg)
        plotter1.plot('Loss_XU', 'labeled_model',
                      'Loss XU', epoch, losses1_vt.avg)
        plotter1.plot('Loss_XU', 'unlabeled_model',
                      'Loss_XU', epoch, losses2_vt.avg)

        print("Time - {}".format((start_time-end)))
        print()
        print("Loss - Epoch {} - Training loss: {}".format(e,
                                                              losses1.avg))
        print("loss_x - Epoch {} - Training loss: {}".format(e,
                                                              losses1_vt.avg))
        print("loss_u - Epoch {} - Training loss: {}".format(e,
                                                              losses2_vt.avg))
        print("     WS - Epoch {} - Training loss: {}".format(e,
                                                              ws.avg))
        print()

    ## END OF TRAINING FUNCTION ##

def accuracy(output, target, topk=(1,)):
    """Computes the precision for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(device, model, mt_model, test_loader, epoch):
    """ Test is used to validate the training of the model on unseen data
    this method takes both models and the loader and runs a series of accuracy tests """
    losses1 = utils.AverageMeter()
    losses2 = utils.AverageMeter()
    criterion = nn.NLLLoss()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    ## Flip models to evaluation mode ##
    model.eval()
    mt_model.eval()
    ## Zero out losses to start ##
    test_loss1 = 0
    test_loss2 = 0
    correct1 = 0
    correct2 = 0
    ## Turn of gradient descent in testing ##
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.shape[0], -1)
            input_var = torch.autograd.Variable(data)
            mt_input = torch.autograd.Variable(data)
            target_var = torch.autograd.Variable(target)
            target_var_up = torch.max(target_var, 1)[1]

            # Get y' from the model
            output1 = model(data)
            output2 = mt_model(data)

            # Check y' against y
            test_loss1 += F.nll_loss(output1, target_var_up,
                                     reduction='sum').item()
            test_loss2 += F.nll_loss(output2, target_var_up,
                                     reduction='sum').item()

            pred1 = output1.argmax(dim=1, keepdim=True)
            pred2 = output2.argmax(dim=1, keepdim=True)

            correct1 += pred1.eq(target_var_up.view_as(pred1)).sum().item()
            correct2 += pred2.eq(target_var_up.view_as(pred2)).sum().item()

        test_loss1 /= len(test_loader.dataset)
        test_loss2 /= len(test_loader.dataset)

        accuracy1 = 100. * correct1 / len(test_loader.dataset)
        accuracy2 = 100. * correct2 / len(test_loader.dataset)

        print('\nStudent test Set: AVG Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss1, correct1, len(test_loader.dataset),
            100. * correct1 / len(test_loader.dataset)))
        print('Teacher test Set: AVG Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss2, correct2, len(test_loader.dataset),
            100. * correct2 / len(test_loader.dataset)))
        #plotter.plot('Validation Loss', 'val', 'Class Loss', epoch, losses1.avg)
        plotter1.plot('Accuracy', 'Student Validation',
                      'Model Accuracy', epoch, accuracy1)
        plotter1.plot('Accuracy', 'Teacher Validation',
                      'Model Accuracy', epoch, accuracy2)
        
        ## END OF TESTING FUNCTION ##

# Import data from specified location #
# Get all labeled and unlabeled in one function #
label_loader, unlabel_loader = dataset.get_unlabelled_data('Data/ntuple_merged_11.h5', 250)

## ARG ooptions ##
envs0 = "Test 1"
envs1 = "Test 2"
envs2 = "Test 3"
envs3 = "Test 4"

## ENV for data/loss/accuracy visualization ##
environments = [envs0, envs1, envs2, envs3]

test_set, test_loader = dataset.get_test_data('Data/ntuple_merged_11.h5')


# Get visdom ready to go #
global plotter1
plotter1 = utils.VisdomLinePlotter(env_name=environments[args.env])

"""Normalizing data with Gaussian noise"""
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                AddGaussianNoise(0., 1.)
                                ])

### Input sizes to be used for the models ###
input_size = 27
hidden_sizes = [256, 128, 64, 64, 64, 32]
output_size = 2
global_step = 0

input_nodes5_1 = [256, 256, 256, 256, 256, 256]
input_nodes5_2 = [256, 256, 512, 512, 256, 128]
input_nodes5_3 = [512, 512, 512, 512, 512, 512]

input_nodes6_1 = [256, 256, 256, 256, 256, 256, 128]
input_nodes6_2 = [256, 256, 512, 512, 512, 256, 128]
input_nodes6_3 = [512, 512, 512, 512, 512, 512, 512]

### Create from arrays ###
model = model_arch.seq_model_5(input_size, input_nodes5_1, output_size, 1, ema=False)
mt_model = model_arch.seq_model_5(input_size, input_nodes5_1, output_size, 1, ema=True)

nnet_arch0 = [512, 512, 256, 256, 128, 128]
nnet_arch1 = [256, 256, 256, 256, 256, 256]
nnet_arch2 = [256, 512, 512, 256, 256, 128]
nnet_arch3 = [512, 512, 512, 512, 256, 128]

nnet_arches = [nnet_arch0, nnet_arch1, nnet_arch2, nnet_arch3]

## Debug printing of model stats ##
print(model)
print(mt_model)

print('     Total Parameters: %2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

#optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
#optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0,
#                         weight_decay=0.01, initial_accumulator_value=0, eps=1e-10)
#optimizer = optim.Adam(model.parameters(), lr=0.003, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)
optimizer = optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
#optimizer = optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0.01)
epochs = args.epochs

for e in range(epochs):
    print('\nEpoch: [%d | %d] LR: %f' % (e + 1, args.epochs, args.learning_rate))
    start_time = time.time()
    running_loss = 0
    train(label_loader, unlabel_loader, model, mt_model, optimizer, e, ema_const=0.90)
    test(device, model, mt_model, test_loader, e)
