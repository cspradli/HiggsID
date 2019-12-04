import keras
import numpy as np
import tables
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os.path

### Attempting to get rid of TF and Keras warnings about deprecation ###
import warnings
warnings.simplefilter("ignore")

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


def get_feature_lables(fname, PTWINDOW=True):
    """Sorts through the H5 file to find the feature array, specs array, and label array. Returns the feature array
    in conjunction with the label array """
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
    
    if PTWINDOW:
        ### checking to see if we can automatically throw out candidates if they are not within mass window ###
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

### Using premade training data from CERN ###
if not os.path.isfile('Data/ntuple_merged_10.h5'):
    print("ERROR: data not found")
    exit(1)

feat_arr, label_arr = get_feature_lables('Data/ntuple_merged_10.h5', PTWINDOW=False)

### Model being put together ###
inputs = Input(shape=(numF,), name='Input')
x = BatchNormalization(name='bn_1')(inputs)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(numLabels, activation='softmax')(x)
keras_mod = Model(inputs=inputs, outputs=outputs)
keras_mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(keras_mod.summary())

### Check to make sure we aren't over training ###
early_stopping = EarlyStopping(monitor='val_loss', patience=50)
model_ch = ModelCheckpoint('models/keras_higgs_best.h5', monitor='val_loss', save_best_only=True)
callb = [early_stopping, model_ch]

### Fit the model ###
keras_mod.fit(feat_arr, label_arr, batch_size=1024, epochs=100,
              validation_split=0.2, shuffle=True, callbacks=callb)


########### Testing for fit model #############
### using premade testing dataset from CERN ###
if not os.path.isfile('Data/ntuple_merged_0.h5'):
    print("ERROR: need testing data")
    print("Need ntuple_merged_0.h5")
    exit(0)

feat_arr_test, label_arr_test = get_feature_lables('Data/ntuple_merged_0.h5')
keras_mod.load_weights('models/keras_higgs_best.h5')
predict_arr_test = keras_mod.predict(feat_arr_test)


########### Plot ROC curve for AUC score #############
fpr, tpr, threshold = roc_curve(label_arr_test[:,1], predict_arr_test[:,1])
plt.figure()
plt.plot(tpr, fpr, lw=2.5, label="AUC = {:.2f}%".format(auc(fpr,tpr)*100))
plt.xlabel(r'True positive rate')
plt.ylabel(r'False positive rate')
plt.semilogy()
plt.ylim(0.001,1)
plt.xlim(0,1)
plt.grid(True)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('auc_score.png')
plt.savefig('auc_score.pdf')



