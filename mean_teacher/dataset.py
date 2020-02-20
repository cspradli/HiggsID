import os.path
import h5py
import tables
import numpy as np
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
            #'fj_tau_vertexMass_1',
            #'fj_trackSip2dSigAboveBottom_0',
            #'fj_trackSip2dSigAboveBottom_1',
            #'fj_trackSip2dSigAboveCharm_0',
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


"""if not os.path.isfile('Data/ntuple_merged_10.h5'):
    print("ERROR: data not found")
    exit(1)"""


# dataset = HDF5Dataset('Data/', recursive=True, load_data=False,
#   data_cache_size=4, transform=None)

#feat_arr, label_arr = get_feature_lables('Data/ntuple_merged_10.h5', remove_mass_PTWINDOW=False)
# print(feat_arr)
# print(label_arr)
#train_load = torch.utils.data.DataLoader(dataset=feat_arr, shuffle = True)
# label_arr.append(feat_arr)
#lab_ay = np.append(label_arr, feat_arr)
# print(lab_ay)
