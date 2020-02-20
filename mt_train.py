import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import utils
import time
import numpy as np
import tables
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os.path

from mean_teacher.mean_teacher import AddGaussianNoise
from mean_teacher import dataset
from mean_teacher import loss_functions
from mean_teacher import mean_teacher

import warnings
warnings.simplefilter("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the data from the HDF5 files, return the feature data alongide the
feat_arr, label_arr = dataset.get_feature_lables(
    'Data/ntuple_merged_11.h5', remove_mass_PTWINDOW=False)
test_feat, test_label = dataset.get_feature_lables(
    'Data/ntuple_merged_1.h5', remove_mass_PTWINDOW=False)

print(feat_arr)

print(label_arr)

#### Convert the numpy data to a Torch-ready data type ###
X = Variable(torch.from_numpy(feat_arr).float(), requires_grad=False)
Y = Variable(torch.from_numpy(label_arr).float(), requires_grad=False)

testX = Variable(torch.from_numpy(test_feat).float(), requires_grad=False)
testY = Variable(torch.from_numpy(test_label).float(), requires_grad=False)

### Turn all the data into a Tensor viable dataset AND dataloader for efficiency ###
test_set = data.TensorDataset(testX, testY)
test_loader = data.DataLoader(test_set, batch_size=1024, shuffle=True)

dat_set = data.TensorDataset(X, Y)
dat_loader = data.DataLoader(dat_set, batch_size=64, shuffle=True)


# Get visdom ready to go #
global plotter1
plotter1 = utils.VisdomLinePlotter(env_name='main')


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

    ## Set both models to training mode ##
    model.train()
    mt_model.train()

    for images, labels in dat_loader:

        global_step += 1

        images = images.view(images.shape[0], -1)

        sl = images.shape

        input_var = images  # torch.autograd.Variable(images)
        mt_input = images  # torch.autograd.Variable(images)
        target_var = labels  # torch.autograd.Variable(labels)

        output = model(input_var)
        with torch.no_grad():
            output1 = mt_model(mt_input)

        output_lab = output[:sl[0]]
        output1_lab = output1[:sl[0]]

        loss_vt = criterion(output_lab, torch.max(target_var, 1)[1])
        loss_mt_vt = criterion_kl(output, output1)

        mt_out = mt_model(input_var)
        model_out = model(mt_input)

        loss = criterion(model_out, torch.max(target_var, 1)[1])
        loss_mt = criterion(mt_out, torch.max(target_var, 1)[1])

        losses1.update(loss.data.cpu().numpy(), labels.size(0))
        losses2.update(loss_mt.data.cpu().numpy(), labels.size(0))
        losses1_vt.update(loss.data.cpu().numpy(), labels.size(0))
        losses2_vt.update(loss_mt.data.cpu().numpy(), labels.size(0))
        ## Get MSE loss ##
        #cl_loss = consistency_criterion(model_out, labels)
        #mt_loss = consistency_criterion(mt_out, labels)

        optimizer.zero_grad()
        loss.backward()
        loss_vt.backward()
        # loss_mt.backward()
        optimizer.step()

        mean_teacher.update_mt(model, mt_model, ema_const, global_step)

        end = time.time()
        run_loss += loss.item()
        run_loss_mt += loss_mt.item()
        run_loss_vt += loss.item()
        run_loss_mt_vt += loss_mt.item()

    else:
        plotter1.plot('Loss', 'student', 'Model Loss', epoch, losses1.avg)
        plotter1.plot('Loss', 'teacher', 'Model Loss', epoch, losses2.avg)
        plotter1.plot('Loss_vt', 'student_vt',
                      'Model Loss_vt', epoch, losses1_vt.avg)
        plotter1.plot('Loss_vt', 'teacher_vt',
                      'Model Loss_vt', epoch, losses2_vt.avg)
        #plotter1.set_text('Log Loss', "Student - Epoch {} - Training loss: {}".format(e, run_loss/len(trainloader)))
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


"""Use this data until figure out other data problem"""
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                AddGaussianNoise(0., 1.)
                                ])

"""
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
valset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
"""
"""model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], hidden_sizes[2]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[2], hidden_sizes[3]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[3], hidden_sizes[4]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[4], output_size),
    nn.Softmax(dim=1)
)"""
### Input sizes to be used for the models ###
input_size = 27
hidden_sizes = [256, 128, 64, 64, 64, 32]
output_size = 2
global_step = 0
"""
model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], hidden_sizes[2]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[2], hidden_sizes[3]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[3], hidden_sizes[4]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[4], output_size),
    nn.Softmax(dim=1)
)

mt_model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], hidden_sizes[2]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[2], hidden_sizes[3]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[3], hidden_sizes[4]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[4], output_size),
    nn.Softmax(dim=1)
)"""

model = nn.Sequential(nn.Linear(23, 128),
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

print(model)
print(mt_model)

for param in mt_model.parameters():
    param.detach_()

#optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0,
                          weight_decay=0.01, initial_accumulator_value=0, eps=1e-10)
#optimizer = optim.Adam(model.parameters(), lr=0.003, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)
#optimizer = optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
#optimizer = optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0.01)
epochs = 50

for e in range(epochs):
    start_time = time.time()
    running_loss = 0
    train(dat_loader, model, mt_model, optimizer, e, ema_const=0.95)
    test(device, model, mt_model, test_loader, e)
