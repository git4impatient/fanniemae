# 
from __future__ import print_function
import os, sys
import time
import seaborn as sns
import pandas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as ks
from keras import Sequential 
from keras.layers import Dense
from keras.callbacks import TensorBoard 
from sklearn.metrics import confusion_matrix
## Fannie Mae failed loan prediction
## Model to predict if a loan will have foreclosure costs
# keras requires python 2  / get tensorflow import error
#classifier.summary()
#print(classifier.metrics()

# save the classifier to deploy
#classifier.save("kerasLoanClassifier.keras.save")

from keras.models import load_model

model = load_model("kerasLoanClassifier.keras.save")

def loanofficer(Z):
  loaneval = (model.predict(Z)>.65)
  return loaneval[0][0]

x = np.array([1.0, 2.0, 3.0])

#|label|channel|
#intrate|
#loanamt|
#loan2val|
#numborrowers|
#itscore|
#fcl_costs|loc_state|
inputvals=""
myvals=[ 0.600,0.00633,1.0824742,0.166666,0.62885,0.0318,3.0]


def loanplease(args):
   x = np.reshape (np.array (myvals),(-1,7))
   return loanofficer(x)

# loanplease(myvals)
# print(x.shape) 