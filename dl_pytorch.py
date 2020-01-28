import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
#from torchvision import datasets, transforms
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import tables
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os.path
import warnings
warnings.simplefilter("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 27 features to train off of
features = ['fj_jetNTracks',
            'fj_nSV',
            'fj_tau0_trackEtaRel_0',
            'fj_tau0_trackEtaRel_1',
            'fj_tau0_trackEtaRel_2',
            'fj_tau1_trackEtaRel_0',
            'fj_tau1_trackEtaRel_1',
            'fj_tau1_trackEtaRel_2',
            'fj_tau_flightDistance2dSig_0',
            'fj_tau_flightDistance2dSig_1',
            'fj_tau_vertexDeltaR_0',
            'fj_tau_vertexEnergyRatio_0',
            'fj_tau_vertexEnergyRatio_1',
            'fj_tau_vertexMass_0',
            'fj_tau_vertexMass_1',
            'fj_trackSip2dSigAboveBottom_0',
            'fj_trackSip2dSigAboveBottom_1',
            'fj_trackSip2dSigAboveCharm_0',
            'fj_trackSipdSig_0',
            'fj_trackSipdSig_0_0',
            'fj_trackSipdSig_0_1',
            'fj_trackSipdSig_1',
            'fj_trackSipdSig_1_0',
            'fj_trackSipdSig_1_1',
            'fj_trackSipdSig_2',
            'fj_trackSipdSig_3',
            'fj_z_ratio']

specs = ['fj_sdmass',
         'fj_pt'
        ]
labels = ['fj_isQCD*sample_isQCD',
          'fj_isH*fj_isBB'
         ]

numF = len(features)
numSpecs = len(specs)
numLabels = len(labels)


def get_feature_lables(fname, remove_mass_PTWINDOW=True):
    file = tables.open_file(fname, 'r')
    numJets = getattr(file.root, features[0]).shape[0]
    feat_arr = np.zeros((numJets, numF))
    spec_arr = np.zeros((numJets, numSpecs))
    lab_arr = np.zeros((numJets, numLabels))

    for (i, feat) in enumerate(features):
        feat_arr[:,i] = getattr(file.root, feat)[:]

    for (i, spec) in enumerate(specs):
        spec_arr[:,i] = getattr(file.root, spec)[:]

    for (i, label) in enumerate(labels):
        prods = label.split('*')
        prod0 = prods[0]
        prod1 = prods[1]
        fact0 = getattr(file.root, prod0)[:]
        fact1 = getattr(file.root, prod1)[:]
        lab_arr[:,i] = np.multiply(fact0, fact1)
    
    if remove_mass_PTWINDOW:
        feat_arr = feat_arr[(spec_arr[:,0] > 40)
                         & (spec_arr[:,0] < 200) 
                         & (spec_arr[:,1] > 300) 
                         & (spec_arr[:,1] < 2000)]
        lab_arr = lab_arr[(spec_arr[:,0] > 40)
                         & (spec_arr[:,0] < 200)
                         & (spec_arr[:,1] > 300)
                         & (spec_arr[:,1] < 2000)]
    
    feat_arr = feat_arr[np.sum(lab_arr, axis=1)==1]
    lab_arr = lab_arr[np.sum(lab_arr, axis=1)==1]

    file.close()
    return feat_arr, lab_arr



if not os.path.isfile('Data/ntuple_merged_10.h5'):
    print("ERROR: data not found")
    exit(1)

"""dataset = HDF5Dataset('Data/', recursive=True, load_data=False, 
   data_cache_size=4, transform=None)"""

feat_arr, label_arr = get_feature_lables('Data/ntuple_merged_10.h5', remove_mass_PTWINDOW=False)
#print(feat_arr)
#print(label_arr)
#train_load = torch.utils.data.DataLoader(dataset=feat_arr, shuffle = True)
#label_arr.append(feat_arr)
lab_ay = np.append(label_arr, feat_arr)
#print(lab_ay)

#X = Variable(torch.from_numpy(feat_arr).float(), requires_grad=False)
#Y = Variable(torch.from_numpy(label_arr).float(), requires_grad=False)

#print(X.size())
#print(Y.size())

#input_size = 16
#hidden_sizes = [256, 128, 64, 64, 64, 32]
#output_size = 2

"""Use this data until figure out other data problem"""
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
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

print(model)

criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

epochs = 5

for e in range(epochs):

    running_loss = 0
    for images, labels in trainloader:


        images = images.view(images.shape[0],-1)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))




correct_count, all_count = 0, 0
for im,labels in valloader:
  for i in range(len(labels)):
    candidate = im[i].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(candidate)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = im.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

