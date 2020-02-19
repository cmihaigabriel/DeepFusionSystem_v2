'''
Created on Jul 9, 2019

@author: cmihaigabriel
@version: v2.1

@change: v1.0 used in 2019 SI IJCV
@change: v2.0 used in 2020 ICMR
@change: v2.1 used in 2020 ECCV
'''

import numpy as np
from LateFusionFileHandler import FileHandler

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization  as bn
from keras.layers import Input, Dense, multiply, Convolution1D, Flatten, AveragePooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from keras import optimizers
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold
from SimMatrixCreator import SimMatrixCreator

def runFusion(no_layers = -1, no_neurons = -1, database = -1, bn_active = 0, kfolds = -1, repeatfolds = -1, gpudevice = "-1", attnLayer = 0, convLayer = 0):
    serializer = FileHandler()
    SetupFileHandler(serializer, database)
    
    serializer.loadFiles()
    
    #Scores from the loaded files (columns = runs, rows = samples) - not sorted'
    input = serializer.getScoresNormalizedSorted()
    labels = serializer.getGtPredictions()

    if convLayer > 0:
        matrixCreator = SimMatrixCreator()

    validation_mask = np.zeros((10,10))
    finalMetricFolds = []

    ###########################
    # running the MLP algorithm

    # -*- coding: utf-8 -*-

    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # suppress CPU msg
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpudevice  # selecting GPU device
    #os.environ["CUDA_VISIBLE_DEVICES"] = "" #run in CPU mode

    class Settings(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.Metric = []
            self.maxMetric = 0.0
            self.maxSecMetric = 0.0
            if database in [4,5]:
                self.maxMetric = 1.0
        
        def on_train_end(self, logs={}):
            print('Max Metric = ' + str(self.maxMetric) + ' Max Second Metric = ' + str(self.maxSecMetric))
            finalMetricFolds.append([self.maxMetric, self.maxSecMetric])
            avgsofar = np.mean(np.array(finalMetricFolds), axis=0)
            print('AVG so far is: ' + str(avgsofar))
            return

        def on_epoch_begin(self, epoch, logs={}):
            return

        def on_epoch_end(self, epoch, logs={}):
            thisloss = logs.get('loss')
            print ('loss = ' + str(thisloss))
            self.losses.append(thisloss)
            y_pred = self.model.predict(self.validation_data[0], batch_size=None, verbose=1, steps=1)
            self.loss_mapper(self.losses, y_pred, thisloss)
            return y_pred

        def loss_mapper(self, losses, y_pred, thisloss):
            if database in [0,1,2]:
                self.Metric.append(serializer.saveFiles(y_pred[:,0], 'MLPS', validation_mask))
                print('Map:', self.Metric[-1:])
                curmetric = float(self.Metric[len(self.Metric) - 1])
                if self.maxMetric < curmetric:
                    self.Metric.append(serializer.saveFiles(y_pred[:,0], 'MLPS_max', validation_mask))
                    self.maxMetric = curmetric
            elif database in [4,5]:
                curmetric = thisloss
                if self.maxMetric > curmetric: #it's > because it's MSE
                    self.maxMetric = curmetric
                    self.maxSecMetric = serializer.calcPearsonCC(y_pred,y_valid_cv)
            elif database in [6]:
                y_class = []
                thresh = serializer.calcIoUThresh(y_pred)
                for i in range(len(y_pred)):
                    if (y_pred[i] < thresh):
                        y_class.append(0)
                    else:
                        y_class.append(1)
                curmetric = serializer.calcIoU(np.array(y_class), y_valid_cv)
                cursecmetric = serializer.calcIoUPerSample(np.array(y_class), y_valid_cv, validation_mask)
                print('Curmetric is ' + str(curmetric) + " CurSecmetric is " + str(cursecmetric))
                if self.maxMetric < curmetric:
                    self.maxMetric = curmetric
                if self.maxSecMetric < cursecmetric:
                    self.maxSecMetric = cursecmetric
                
            
            return

            
        def on_batch_begin(self, batch, logs={}):
            return

        def on_batch_end(self, batch, logs={}):
            return
    
    def save_model(name_weights, patience_lr):
        mcp_save = ModelCheckpoint(filepath='./models/' + name_weights, save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
        return [mcp_save, reduce_lr_loss]

    def load_data_kfold(x_train, y_train, k, rep = 1, database = 0):
        #folds = list(StratifiedKFold(n_splits=k, shuffle=False, random_state=42).split(x_train, y_train))
        #folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(x_train, y_train))
        if database in [0,1,2,6]:
            folds = list(RepeatedStratifiedKFold(n_splits=k, n_repeats = rep, random_state=28).split(x_train, y_train))
        if database in [4,5]:
            folds = list(RepeatedKFold(n_splits=k, n_repeats = rep, random_state=28).split(x_train, y_train))
        return folds, x_train, y_train

    def get_number_classes(database):
        if database in [0,1,2,6]:
            return 1
        if database in [4,5]:
            return 1
        return 1

    def get_model_seq():
        model = Sequential()

        if convLayer > 0:
            model.add(Convolution1D(convLayer, 3, border_mode='same', input_shape=(x_train_cv.shape[1], x_train_cv.shape[2])))
            model.add(AveragePooling1D(pool_size=3))
            #model.add(Conv1D(1, 3, padding='same', input_shape=(x_train_cv.shape[1], x_train_cv.shape[2])))
            model.add(Flatten())

        model.add(Dense(no_neurons, activation='relu', input_shape=(x_train_cv.shape[1],)))
        if bn_active:
            model.add(bn())

        for i in range(no_layers - 2):
            model.add(Dense(no_neurons, activation='relu'))
            if bn_active:
                model.add(bn())

        model.add(Dense(no_neurons, activation='relu'))
        model.add(Dense(num_classes, activation='sigmoid'))

        model.summary()
        adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
        if database in [0,1,2,6]:
            model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        elif database in [4,5]:
            model.compile(optimizer=adam, loss='mean_squared_error')
        else:
            model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def get_model_attntype():
        input_layer = Input(shape=(input.shape[1],))
        dense = Dense(no_neurons, activation='relu')(input_layer)
        if bn_active:
            batchnorm = bn()(dense)

            for i in range(no_layers - 2):
                dense = Dense(no_neurons, activation='relu')(batchnorm)
                batchnorm = bn()(dense)

            dense = Dense(no_neurons, activation='relu')(batchnorm)
        else:
            for i in range(no_layers - 2):
                dense = Dense(no_neurons, activation='relu')(dense)
            dense = Dense(no_neurons, activation='relu')(dense)
        conf_layer = Dense(num_classes, activation='sigmoid')(dense)
        attention_probs = Dense(num_classes, activation='softmax', name='attention_probs')(input_layer)
        attention_mul = multiply([conf_layer, attention_probs], name='attention_mul')

        model = Model(input=[input_layer], output=attention_mul)
        adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
        if database in [0,1,2,6]:
            model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        elif database in [4,5]:
            model.compile(optimizer=adam, loss='mean_squared_error')
        else:
            model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()
        return model


    folds, x_train, y_train = load_data_kfold(input, labels, kfolds, repeatfolds, database)
    
    print(x_train.shape[0], 'train samples')
    num_classes = get_number_classes(database)

    for j, (train_idx, val_idx) in enumerate(folds):
    
        print('\nFold ',j)
        x_train_cv = x_train[train_idx]
        y_train_cv = y_train[train_idx]
        x_valid_cv = x_train[val_idx]
        y_valid_cv= y_train[val_idx]
        validation_mask = val_idx

        if convLayer > 0:
            (x_train_cv, x_valid_cv) = matrixCreator.getMatrix_C1D_4N4S(x_train_cv, x_valid_cv)
    
        name_weights = "final_model_fold" + str(j) + "_weights.h5"
        # prepare callback
        no_epochs = 20
        if attnLayer == 0:
            if convLayer > 0:
                no_epochs = 50
            model = get_model_seq()
        else:
            model = get_model_attntype()
            no_epochs = 50
        settings = Settings()

        train = model.fit(x_train_cv, y_train_cv,
                            batch_size=64,
                            epochs=no_epochs,
                            verbose=0,
                            validation_data=(x_valid_cv, y_valid_cv),
                            callbacks=[settings])
        K.clear_session()

    return np.mean(np.array(finalMetricFolds), axis=0)
    

def SetupFileHandler(ser, type):
    '''
    Calls the serializer setup
    @param ser: serializer object
            type: type of dataset
            
    type values:
        0 - INT2017.Video 
        1 - INT2017.Image
        2 - VSD2015
        3 - MEDCapt2019 ???
        4 - EMOAroval2018.Arousal
        5 - EMOAroval2018.Valence
        6 - EMOFear2018
    '''

    #####INT2017.Video
    if (type == 0):
        ser.setup('../DataSources/MediaEvalInterestingness/INT2017.Video/SortedRuns/2017Movies',    #runfolder
                    '../FusionCache/FusionResults0/',    #output folder - no longer used
                    'conINT2017Video',      #output filenames - no longer used
                    '../DataSources/MediaEvalInterestingness/INT2017.Video/SortedRuns/GTFiles/testset-2017-video.qrels',    #gt file 
                    '../DataSources/MediaEvalInterestingness/INT2017.Video/treceval/trec_eval',       #eval tool path
                    2435,   #number of samples
                    42,     #number of runs
                    1,      #metric used
                    0)      #competition code
    #####INT2017.Image
    if (type == 1):
        ser.setup('../DataSources/MediaEvalInterestingness/INT2017.Image/SortedRuns/2017Image', 
                    '../FusionCache/FusionResults1/',
                    'conINT2017Image',
                    '../DataSources/MediaEvalInterestingness/INT2017.Image/SortedRuns/GTFiles/testset-2017-image.qrels', 
                    '../DataSources/MediaEvalInterestingness/INT2017.Image/treceval/trec_eval',
                    2435, 
                    33,
                    1,
                    1)
    #####VSD2015
    if (type == 2):
        ser.setup('../DataSources/MediaEvalViolence/VSD2015/SortedRuns',
                    '../FusionCache/FusionResults2/',
                    'conVSD2015',
                    '../DataSources/MediaEvalViolence/VSD2015/GT/violence.qrel',
                    '../DataSources/MediaEvalViolence/VSD2015/EvalTool/trec_eval',
                    4756,
                    48,
                    0,
                    2)
    #####MEDCapt2019
    if (type == 3):
        ser.setup('../DataSources/ImageCLEFMedical/MEDCapt2019',
                  '../FusionCache/FusionResults3/',
                  'conMEDCapt2019',
                  '../DataSources/ImageCLEFMedical/MEDCapt2019/GT/ConceptDetection_GT.txt',
                  '../DataSources/ImageCLEFMedical/MEDCapt2019/EvalTool/evaluate-f1.py',
                  10000,    #number of samples
                  63,       #number of runs
                  2,        #metric used
                  3)        #competition code
    #####EMOAroval2019.Arousal
    if (type == 4):
        ser.setup('../DataSources/MediaEvalEmotions/EMOAroval2018/SortedRuns',
                  '../FusionCache/FusionResults4/',
                  'conEMOAroval2019Arousal',
                  '../DataSources/MediaEvalEmotions/EMOAroval2018/GT/MediaEvalEmo-Valence-Arousal.txt',
                  '../DataSources/MediaEvalEmotions/EMOAroval2018/EvalTool/compute_results_subtask1.py',
                  32196,    #number of samples
                  30,       #number of runs
                  3,        #metric used
                  4)        #competition code
    #####EMOAroval2019.Valence
    if (type == 5):
        ser.setup('../DataSources/MediaEvalEmotions/EMOAroval2018/SortedRuns',
                  '../FusionCache/FusionResults5/',
                  'conEMOAroval2019Valence',
                  '../DataSources/MediaEvalEmotions/EMOAroval2018/GT/MediaEvalEmo-Valence-Arousal.txt',
                  '../DataSources/MediaEvalEmotions/EMOAroval2018/EvalTool/compute_results_subtask1.py',
                  32196,    #number of samples
                  30,       #number of runs
                  3,        #metric used
                  5)        #competition code
    #####EMOFear2019
    if (type == 6):
        ser.setup('../DataSources/MediaEvalEmotions/EMOFear2018/SortedRuns',
                  '../FusionCache/FusionResults6/',
                  'conEMOFear2019',
                  '../DataSources/MediaEvalEmotions/EMOFear2018/GT/fear_gt_mod2.txt',
                  '../DataSources/MediaEvalEmotions/EMOFear2018/EvalTool/compute_results_subtask2.py',
                  32146,    #number of samples
                  18,       #number of runs
                  4,        #metric used
                  6)        #competition code
