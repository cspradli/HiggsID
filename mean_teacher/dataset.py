import os
import os.path
import h5py
import tables
import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import args_util
args = args_util.get_args()
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
    """ Takes in a path to a HDF5 file, returns with numpy arrays of 27 features with labels """

    file = tables.open_file(fname, 'r')
    numJets = getattr(file.root, features[0]).shape[0]
    feat_arr = np.zeros((numJets, numF))
    spec_arr = np.zeros((numJets, numSpecs))
    lab_arr = np.zeros((numJets, numLabels))

    for (i, feat) in enumerate(features):
        feat_arr[:, i] = getattr(file.root, feat)[:]

    for (i, spec) in enumerate(specs):
        spec_arr[:, i] = getattr(file.root, spec)[:]

    for (i, label) in enumerate(labels):
        prods = label.split('*')
        prod0 = prods[0]
        prod1 = prods[1]
        fact0 = getattr(file.root, prod0)[:]
        fact1 = getattr(file.root, prod1)[:]
        lab_arr[:, i] = np.multiply(fact0, fact1)

    if remove_mass_PTWINDOW:
        feat_arr = feat_arr[(spec_arr[:, 0] > 40)
                            & (spec_arr[:, 0] < 200)
                            & (spec_arr[:, 1] > 300)
                            & (spec_arr[:, 1] < 2000)]
        lab_arr = lab_arr[(spec_arr[:, 0] > 40)
                          & (spec_arr[:, 0] < 200)
                          & (spec_arr[:, 1] > 300)
                          & (spec_arr[:, 1] < 2000)]

    feat_arr = feat_arr[np.sum(lab_arr, axis=1) == 1]
    lab_arr = lab_arr[np.sum(lab_arr, axis=1) == 1]

    file.close()
    return feat_arr, lab_arr



def get_labelled_data(input1):
    """ Function to get all necessary data from the HDF5 data files, then turn them all into
    PyTorch datasets """
    # Get the data from the HDF5 files, return the feature data alongide the

    if not os.path.isfile(input1):
        print("ERROR: training data not found")
        exit(1)

    feat_arr, label_arr = get_feature_lables(
        input1, remove_mass_PTWINDOW=False)

    print(feat_arr.shape)
    print(label_arr.shape)

    #### Convert the numpy data to a Torch-ready data type ###
    X = Variable(torch.from_numpy(feat_arr).float(), requires_grad=False)
    Y = Variable(torch.from_numpy(label_arr).float(), requires_grad=False)

    dat_set = data.TensorDataset(X, Y)
    dat_loader = data.DataLoader(dat_set, batch_size=1024, shuffle=True)

    return dat_set, dat_loader



def get_test_data(input2):
    """ Function to get all necessary data from the HDF5 data files, then turn them all into
    PyTorch datasets """
    # Get the data from the HDF5 files, return the feature data alongide the

    if not os.path.isfile(input2):
        print("ERROR: testing data not found")
        exit(1)

    test_feat, test_label = get_feature_lables(
        input2, remove_mass_PTWINDOW=False)

    testX = Variable(torch.from_numpy(test_feat).float(), requires_grad=False)
    testY = Variable(torch.from_numpy(test_label).float(), requires_grad=False)

    ### Turn all the data into a Tensor viable dataset AND dataloader for efficiency ###
    test_set = data.TensorDataset(testX, testY)
    test_loader = data.DataLoader(test_set, batch_size=64, shuffle=True)


    return test_set, test_loader


def get_unlabelled_data(input1, amount_l):
    """ Function to get all necessary (labelled AND unlabelled) data from the HDF5 data files, then turn them all into
    PyTorch datasets """

    # Check to see if data is present #
    if not os.path.isfile(input1):
        print("ERROR: training data not found")
        exit(1)

    

    # Get the data from the HDF5 files, return the feature data alongide the
    x_features, y_labels = get_feature_lables(
        input1, remove_mass_PTWINDOW=True)
    print(x_features.shape)
    print(y_labels.shape)

    #u_y = np.zeros(u_y.shape)
    #uX = Variable(torch.from_numpy(u_x).float(), requires_grad=False)
    #uY = Variable(torch.from_numpy(u_y).float(), requires_grad=False)

    #u_set = data.TensorDataset(uX, uY)
    #u_loader = data.DataLoader(u_set, batch_size=1024, shuffle=True)

    #### Convert the numpy data to a Torch-ready data type ###
    X = Variable(torch.from_numpy(x_features).float(), requires_grad=False)
    Y = Variable(torch.from_numpy(y_labels).float(), requires_grad=False)

    # The Master dataset #
    dat_set = data.TensorDataset(X, Y)

    # Seperate out the labeled from unlabeled via a random split given amount of labeled data we need #
    label_set, unlabel_set = data.random_split(dat_set, [amount_l, (len(dat_set)-amount_l)])
    label_loader = data.DataLoader(label_set, args.batch_size, shuffle=True, drop_last=True)
    unlabel_loader = data.DataLoader(unlabel_set, args.batch_size, shuffle=True, drop_last=True)

    return label_loader, unlabel_loader

