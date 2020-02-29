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

warnings.simplefilter("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = args_util.get_args()

#Debug printing#
print(args.epochs)
print(args.batch_size)
print(args.percent_unlabeled)


def train(train_loader, model, mt_model, optimizer, epoch, ema_const=0.95):
    global global_step
    losses1 = utils.AverageMeter()
    losses2 = utils.AverageMeter()
    losses1_vt = utils.AverageMeter()
    losses2_vt = utils.AverageMeter()
    ## Choose between loss criterion ##
    criterion_ce = nn.CrossEntropyLoss(size_average=False)
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

    for i, (images, labels) in enumerate(dat_loader):
        


        global_step += 1

        images = images.view(images.shape[0], -1)

        sl = images.shape
        minibatch_size = len(labels)

        #input_var = images  # torch.autograd.Variable(images)
        #mt_input = images  # torch.autograd.Variable(images)
        #target_var = labels  # torch.autograd.Variable(labels)
        input_var = Variable(images)
        mt_input = Variable(images)
        target_var = Variable(labels)

        ## Compute guessed labels for unlabelled##
        with torch.no_grad():
            outputs_u = mt_model(mt_input)
            outputs_u2 = mt_model(input_var)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/0.5)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        
        
        output = model(input_var)


        output_lab = output[:sl[0]]
        #output1_lab = output1[:sl[0]]

        loss_vt = criterion(output_lab, torch.max(target_var, 1)[1]) / minibatch_size
        loss_mt_vt = criterion_kl(output, targets_u) / minibatch_size
        #print(loss_mt_vt)
        mt_out = mt_model(input_var)
        model_out = model(mt_input)

        loss = criterion(model_out, torch.max(target_var, 1)[1]) / minibatch_size
        loss_mt = criterion(mt_out, torch.max(target_var, 1)[1]) /minibatch_size
        #print(loss.item())
        losses1.update(loss.data.cpu().numpy(), labels.size(0))
        losses2.update(loss_mt.data.cpu().numpy(), labels.size(0))
        losses1_vt.update(loss_vt.data.cpu().numpy(), labels.size(0))
        losses2_vt.update(loss_mt_vt.data.cpu().numpy(), labels.size(0))
        ## Get MSE loss ##
        #cl_loss = consistency_criterion(model_out, labels)
        #mt_loss = consistency_criterion(mt_out, labels)

        optimizer.zero_grad()
        loss.backward()
        loss_vt.backward()
        #loss_mt.backward()
        optimizer.step()

        mean_teacher.update_mt(model, mt_model, ema_const, global_step)

        end = time.time()
        run_loss += loss.item()
        run_loss_mt += loss_mt.item()
        run_loss_vt += loss_vt.item()
        run_loss_mt_vt += loss_mt_vt.item()

    else:
        plotter1.plot('Loss', 'student', 'Model Loss', epoch, losses1.avg)
        plotter1.plot('Loss', 'teacher', 'Model Loss', epoch, losses2.avg)
        plotter1.plot('Loss_vt', 'student_vt',
                      'Model Loss_vt', epoch, losses1_vt.avg)
        plotter1.plot('Loss_vt', 'teacher_vt',
                      'Model Loss_vt', epoch, losses2_vt.avg)
        #plotter1.set_text('Log Loss', "Student - Epoch {} - Training loss: {}".format(e, run_loss/len(trainloader)))
        print("Time - {}".format((start_time-end)))
        print("Student - Epoch {} - Training loss: {}".format(e,
                                                              run_loss/len(dat_loader)))
        print("Teacher - Epoch {} - Training loss: {}".format(e,
                                                              run_loss_mt/len(dat_loader)))
        print()
        print("Student - Epoch {} - Training loss: {}".format(e,
                                                              run_loss_vt/len(dat_loader)))
        print("Teacher - Epoch {} - Training loss: {}".format(e,
                                                              run_loss_mt_vt/len(dat_loader)))
        print()


def test(device, model, mt_model, test_loader, epoch):
    """ Test is used to validate the training of the model on unseen data
    this method takes both models and the loader and runs a series of accuracy tests """
    losses1 = utils.AverageMeter()
    losses2 = utils.AverageMeter()
    criterion = nn.NLLLoss()

    model.eval()
    mt_model.eval()
    test_loss1 = 0
    test_loss2 = 0
    correct1 = 0
    correct2 = 0

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

            #losses1.update(test_loss1.data.cpu().numpy(), target.size(0))
            #losses2.update(test_loss2.data.cpu().numpy(), target.size(0))

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


# Import data from specified location #
dat_set, dat_loader = dataset.get_labelled_data(
    'Data/ntuple_merged_11.h5')

#dat_loader, unlabeled_loader = dataset.get_unlabelled_data(
#    'Data/ntuple_merged_11.h5', args.percent_unlabeled)



test_set, test_loader = dataset.get_test_data('Data/ntuple_merged_11.h5')

# Get visdom ready to go #
global plotter1
plotter1 = utils.VisdomLinePlotter(env_name='main')

"""Use this data until figure out other data problem"""
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                AddGaussianNoise(0., 1.)
                                ])

### Input sizes to be used for the models ###
input_size = 27
hidden_sizes = [256, 128, 64, 64, 64, 32]
output_size = 2
global_step = 0


#Creat nn from model architectures in mean teacher#
model = model_arch.creat_seq_model()
mt_model = model_arch.creat_seq_model(ema=True)

"""
odel = nn.Sequential(nn.Linear(23, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 2),
                      nn.LogSoftmax(dim=1))

mt_model = nn.Sequential(nn.Linear(23, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 2),
                      nn.LogSoftmax(dim=1))
"""

print(model)
print(mt_model)



#optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
#optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0,
#                         weight_decay=0.01, initial_accumulator_value=0, eps=1e-10)
#optimizer = optim.Adam(model.parameters(), lr=0.003, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)
optimizer = optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
#optimizer = optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0.01)
epochs = args.epochs

for e in range(epochs):
    start_time = time.time()
    running_loss = 0
    train(dat_loader, model, mt_model, optimizer, e, ema_const=0.90)
    test(device, model, mt_model, test_loader, e)
