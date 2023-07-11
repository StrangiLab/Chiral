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
import tf.keras.initializers as Kini
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

dr = '/home/lini/Documents/chiral/models/'

print('Loading input data..')
inData = sio.loadmat('/home/lini/Documents/chiral/data/allInputStructures.mat')
(numStructs,zz,zzz) = inData.shape
testData = np.round(numStructs/10).astype(np.int)
structure = np.zeros((numStructs-testData,21,21))
structure_test = np.zeros((testData,21,21))
structure = inData[:numStructs-testData,:,:]
structure_test = inData[numStructs-testData:,:,:]
print('Input data loaded! : %d'%(numStructs))
print('Dedcation: %d Training, %d test'%(numStructs-testData,testData))

print('Loading output data..')
outData = np.loadtxt('/home/lini/Documents/chiral/data/allOutput_CPL-1.txt')
(numStructsOut,numOut) = outData.shape
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

    nla = int(x[1])
    np1 = int(x[2])
    np2 = int(x[3])
    drop = 0.01 

    structure = Input((21,21))
    
    conv1 = Conv1D(64,2,activation='relu',padding='same')(structure)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    fc1 = Flatten()(conv1)
    
    kernel = Kini.Identity()
    bias = Kini.Zeros()
    conv2 = Conv1D(64,2,activation='relu',padding='same',kernel_initializer=kernel,bias_initializer=bias)(structure)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    fc2 = Flatten()(conv2)

    fc = Add()([fc1,fc2])

    d1 = Dense(np1,activation='relu')(fc)
    den = Dropout(drop)(d1)
    for i in range(nla):
        den = Dense(np1,activation='relu')(den)
        den = Dropout(drop)(den)

    d2 = Dense(np2,activation='relu')(den)
    out = Dense(numOut,activation='none')(out)

    model = Model(structure,out)
    model.compile(optimizer='adam',loss='mse')
    model.summary()        
        
        
    modname = dr+'gammadion-stacked_CNN'+str(num)+'step'
    print('-'*57)
    print(modname)
    print('-'*57)
    with open(modname+'_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    print('Model Summary Saved!')
    num += 1        
    return {'model':model,
            'modname':modname,
            'a':a}

def fit_fcn(x):
    global num
    defn = define_model(x)
    model = defn['model']
    modname = defn['modname']
    
    a = x[4]
    def decay(ep):
        global a
        lr = a/((ep)+1)#simple 1 param 1/sqrt(t) decat
        return lr
    lr = LRS(decay)

    leng = 200
    history = model.fit(structure,output,epochs=leng,batch_size=2,verbose=0,validation_data=(structure_test,ouput_test),callbacks=[decay],workers=2,use_multiprocessing=True)
    model.save(modname)
    (te_loss) = model.evaluate(struct_test,ouput_test)
    print(te_loss)
    
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
    return {'model':model,'modname':modname,'loss':te_loss[0]}

global num
num0 = 0
num=num0
numTrials = 100
numModel = 5
trials = np.ones((numTrials,5))*99999999
for i in range(numTrials):
    nla = np.round(np.random.uniform(0,3)).astype(np.int)
    np1 = np.round(np.random.uniform(200,1000)).astype(np.int)
    np2 = np.round(np1*2).astype(np.int)
    a = np.abs(np.random.normal(0,1E-3))
    x = np.array([nla,np1,np2,a])
    fit = fit_fcn(x)
    trials[i,:] = np.concatenate((x,np.array([fit['loss']])))
    snew = set(trials[:i,4])
    if i<numModel:
        print('Model Saving to Disk...')
        model = fit['model']
        model.save(modname)
        sold = snew
        print('Model Saved!')
    elif trials[i,4]<sold[numModel-1]:
        print('Model Saving to Disk...')
        nrem = np.argmin(np.abs(trials[:,4]-sold[numModel]))
        print('Removing: %s'%('gammadion-stacked_CNN'+str(nrem+num0)+'step'))
        os.remove(dr+'gammadion-stacked_CNN'+str(nrem+num0)+'step')
        model = fit['model']
        model.save(fit['modname'])
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

        
        
        
        
        