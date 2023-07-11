# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 09:16:59 2021

@author: Alpha
"""
import warnings
warnings.filterwarnings('ignore')

#train CNN keras

import numpy as np
import tensorflow as tf
from keras import callbacks
from keras.callbacks import LearningRateScheduler as LRS
from keras.models import Model
from keras.layers import *
from keras import optimizers as opt
import keras.initializers as Kini
import keras.backend as K
import matplotlib.pyplot as plt
import datetime
global date
date = datetime.datetime.now()
import h5py
import scipy.io as sio
# from hyperopt import fmin, tpe, hp, STATUS_OK,Trials
from contextlib import redirect_stdout
import csv
import os
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# dr = '/home/lini/Documents/chiral/models/'
dr = 'C:/Users/Alpha/Documents/andy/'

print('Loading input data..')
# inData = sio.loadmat('/home/lini/Documents/chiral/data/allInputStructures.mat')
inDataDict = sio.loadmat('C:/Users/Alpha/Documents/andy/chiral_models_comsol/2step_gammadion_data/newdata1-17/allInputStructures.mat');
inData = inDataDict['allInput']
# inData = np.take(inData,np.random.permutation(inData.shape[0]),axis=0,out=inData);
(numStructs,zz,zzz) = inData.shape
testData = np.round(numStructs/10).astype(np.int)
structure = np.zeros((numStructs-testData,21,21))
structure_test = np.zeros((testData,21,21))
structure = inData[:numStructs-testData,:,:]
structure_test = inData[numStructs-testData:,:,:]
print('Input data loaded! : %d structures'%(numStructs))
print('Dedcation: %d Training, %d test'%(numStructs-testData,testData))

print('Loading output data..')
# outData = np.loadtxt('/home/lini/Documents/chiral/data/allOutput_CPL-1.txt')
outData = np.loadtxt('C:/Users/Alpha/Documents/andy/chiral_models_comsol/2step_gammadion_data/newdata1-17/allOutput_CPL-1.txt',delimiter=',');
(numStructsOut,numOut) = outData.shape
# outData = np.take(outData,np.random.permutation(outData.shape[0]),axis=0,out=outData);
assert (numStructs == numStructsOut,'\n\nData dimension mismatch!!!!\n\n')
print('Data loaded! : %d'%(numStructsOut))
maxOut = np.max(outData)
minOut = np.min(outData)
outData = (outData-minOut)/(maxOut-minOut)
print('Rescaled Ouput: [%2.2f,%2.2f] --> [%1.1f,%1.1f]'%(minOut,maxOut,np.min(outData),np.max(outData)))
output = outData[:numStructs-testData,:]
output_test = outData[numStructs-testData:,:]
print('Dedcation: %d Training, %d test'%(numStructs-testData,testData))


def define_model(x):
    global num

    nla = int(x[0])
    np1 = int(x[1])
    np2 = int(x[2])
    drop = 0.1

    structure = Input((21,21))
    
    conv1 = Conv1D(64,1,activation='relu',padding='same')(structure)
    conv1 = LeakyReLU(alpha=0.5)(conv1)
    fc1 = Flatten()(conv1)
    
    kernel = Kini.Ones()
    bias = Kini.Zeros()
    conv2 = Conv1D(64,1,activation='relu',padding='same',kernel_initializer=kernel,bias_initializer=bias)(structure)
    conv2 = LeakyReLU(alpha=0.5)(conv2)
    fc2 = Flatten()(conv2)

    fc = Add()([fc1,fc2])

    d1 = Dense(1200,activation='relu')(fc)
    den = Dropout(drop)(d1)
    for i in range(2):
        den = Dense(1000,activation='relu')(den)
        den = Dropout(drop)(den)

    d2 = Dense(1500,activation='relu')(den)
    out = Dense(numOut,activation=None)(d2)

    model = Model(structure,out)
    model.compile(optimizer='adam',loss='mse')
    model.summary()        
        
        
    modname = dr+'gammadion-stacked_CNN-'+str(num)+'step'
    print('-'*57)
    print(modname)
    print('-'*57)
    num += 1        
    return {'model':model,
            'modname':modname,
            'a':a}

def fit_fcn(x):
    global num
    defn = define_model(x)
    model = defn['model']
    modname = defn['modname']
    
    a = x[3]
    def decay(ep):
        global a
        lr = a/((ep)+1)#simple 1 param 1/sqrt(t) decat
        return lr
    lr = LRS(decay)

    leng = 15
    history = model.fit(structure,output,epochs=leng,batch_size=2,verbose=0,validation_data=(structure_test,output_test),callbacks=[lr])
    model.save(modname)
    (te_loss) = model.evaluate(structure_test,output_test)
    print('RMSE: %1.5f'%(np.sqrt(te_loss)))
    
    s1 = model.predict(np.array([structure[0,:,:]])).reshape((50,50)).T
    s1r = output[0,:].reshape((50,50)).T
    plt.pcolormesh(s1)
    plt.show()
    plt.pcolormesh(s1r)
    plt.show()
    s1 = model.predict(np.array([structure[1,:,:]])).reshape((50,50)).T
    s1r = output[1,:].reshape((50,50)).T
    plt.pcolormesh(s1)
    plt.show()
    plt.pcolormesh(s1r)
    plt.show()
    
    
    # try:
    #     plt.plot(range(len(history.history['lr'])),history.history['lr'])
    #     plt.legend(['LearningRate'])
    #     plt.savefig(modname+'loss_fcn.png', dpi=300)
    # except:
    #     print('PlotLoss fig not saved.\n')
    # try:
    #     np.savetxt(modname+'_saveresults.txt',te_loss)
    #     sio.savemat(modname+'_history.mat',history.history)
    # except:
    #     print('Step Data not saved.\n')
    return {'model':model,'modname':modname,'loss':te_loss}

global num
num0 = 100
num=num0
numTrials = 100
numModel = 5
trials = np.ones((numTrials,5))*99999999
for i in range(numTrials):
    nla = np.round(np.random.uniform(0,3)).astype(np.int)
    np1 = np.round(np.random.uniform(200,500)).astype(np.int)
    np2 = np.round(np1*2).astype(np.int)
    a = np.abs(np.random.normal(0,1E-3))
    x = np.array([nla,np1,np2,a])
    fit = fit_fcn(x)
    trials[i,:] = np.concatenate((x,np.array([fit['loss']])))
    snew = np.sort(trials[:i+1,4])
    if i<numModel:
        print('Model Saving to Disk...')
        model = fit['model']
        modname = fit['modname']
        model.save(modname)
        with open(modname+'_summary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        print('Model Summary Saved!')
        sold = snew
        print('Model Saved!')
    elif trials[i,4]<sold[numModel-1]:
        print('Model Saving to Disk...')
        nrem = np.argmin(np.abs(trials[:,4]-sold[numModel-1]))
        print('Removing: %s'%('gammadion-stacked_CNN-'+str(nrem+num0)+'step'))
        os.remove(dr+'gammadion-stacked_CNN-'+str(nrem+num0)+'step')
        model = fit['model']
        model.save(fit['modname'])
        with open(modname+'_summary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        print('Model Summary Saved!')
        sold = snew
        print('Model Saved!')
    np.savetxt(dr+'results_'+str(num0)+'.txt',trials)
    
np.savetxt(dr+'results_'+str(num0)+'.txt',trials)
print('-'*20)
print('-'*20)
print('-'*20)
print('Best Loss: ',snew[:5])

for i in range(numModel):
    nrem = np.argmin(np.abs(trials[:,4]-sold[numModel]))
    print('----Model #%d: %d----'%(i,nrem))
    print(trials[nrem,:4])
    print('Loss: %2.4e'%(trials[nrem,4]))
    print('-'*20,'\n')
print('-'*20,'EOF''-'*20,)

        
        
        
        
        