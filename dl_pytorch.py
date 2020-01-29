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

#X = Variable(torch.from_numpy(feat_arr).float(), requires_grad=False)
#Y = Variable(torch.from_numpy(label_arr).float(), requires_grad=False)

#feat_arr, label_arr = get_feature_lables('Data/ntuple_merged_10.h5', remove_mass_PTWINDOW=False)

#print(X.size())
#print(Y.size())

#input_size = 16
#hidden_sizes = [256, 128, 64, 64, 64, 32]
#output_size = 2
global_step = 0
global plotter1
plotter1 = utils.VisdomLinePlotter(env_name='Tutorial Plots')
global plotter2
plotter2 = utils.VisdomLinePlotter(env_name='Tutorial Plots')

def train(train_loader, model, mt_model, optimizer, epoch, ema_const = 0.95):
    global global_step
    losses1 = utils.AverageMeter()
    losses2 = utils.AverageMeter()
    ## Choose between loss criterion ##
    criterion = nn.NLLLoss()
    #criterion = nn.CrossEntropyLoss(size_average=False)
    ##Running loss for output ##
    run_loss = 0
    run_loss_mt = 0

    consistency_criterion = loss_functions.softmax_meanse

    ## Set both models to training mode ##
    model.train()
    mt_model.train()
    for images, labels in train_loader:

        global_step += 1

        images = images.view(images.shape[0],-1)

        #input_var = torch.autograd.Variable(images)
        #mt_input = torch.autograd.Variable(images)
        #target_var = torch.autograd.Variable(labels)


        mt_out = mt_model(images)
        model_out = model(images)
        loss = criterion(model_out, labels)
        loss_mt = criterion(mt_out, labels)
        losses1.update(loss.data.cpu().numpy(), labels.size(0))
        losses2.update(loss_mt.data.cpu().numpy(), labels.size(0))
        ## Get MSE loss ##
        #cl_loss = consistency_criterion(model_out, labels)
        #mt_loss = consistency_criterion(mt_out, labels)

        optimizer.zero_grad()
        loss.backward()
        loss_mt.backward()
        optimizer.step()

        mean_teacher.update_mt(model, mt_model, ema_const, global_step)

        end = time.time()
        run_loss += loss.item()
        run_loss_mt += loss_mt.item()

    else:
         plotter1.plot('Loss', 'student', 'Model Loss', epoch, losses1.avg)
         plotter1.plot('Loss', 'teacher', 'Model Loss', epoch, losses2.avg)
         print("Student - Epoch {} - Training loss: {}".format(e, run_loss/len(trainloader)))
         print("Teacher - Epoch {} - Training loss: {}".format(e, run_loss_mt/len(trainloader)))
         print()

    
def test(device, model, mt_model, test_loader):

  losses1 = utils.AverageMeter()
  losses2 = utils.AverageMeter()

  model.eval()
  mt_model.eval()
  test_loss1 = 0
  test_loss2 = 0
  correct1 = 0
  correct2 = 0

  with torch.no_grad():
    
    for data, target in test_loader:
      data = data.view(data.shape[0],-1)
      data, target = data.to(device), target.to(device)
      
      output1 = model(data)
      output2 = mt_model(data)

      test_loss1 += F.nll_loss(output1, target, reduction='sum').item()
      test_loss2 += F.nll_loss(output2, target, reduction='sum').item()

      losses1.update(test_loss1.data.cpu().numpy(), target.size(0))
      losses2.update(test_loss2.data.cpu().numpy(), target.size(0))

      pred1 = output1.argmax(dim=1, keepdim=True)
      pred2 = output2.argmax(dim=1, keepdim=True)

      correct1 += pred1.eq(target.view_as(pred1)).sum().item()
      correct2 += pred2.eq(target.view_as(pred2)).sum().item()

    test_loss1 /= len(test_loader.dataset)
    test_loss2 /= len(test_loader.dataset)

    print('\nStudent test Set: AVG Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss1, correct1, len(test_loader.dataset),
      100. * correct1 / len(test_loader.dataset)))
    print('\nTeacher test Set: AVG Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss2, correct2, len(test_loader.dataset),
      100. * correct2 / len(test_loader.dataset)))
    #plotter.plot('Validation Loss', 'val', 'Class Loss', epoch, losses1.avg)
    #plotter.plot('Accuracy', 'val', 'Class Accuracy', epoch, acc)



"""Use this data until figure out other data problem"""
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                AddGaussianNoise(0.,1.)
                              ])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
valset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)



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

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

mt_model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

print(model)
print(mt_model)

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
epochs = 10

for e in range(epochs):
    start_time = time.time()
    running_loss = 0
    train(trainloader, model, mt_model, optimizer, e, ema_const=0.95)


test(device, model, mt_model, valloader)




