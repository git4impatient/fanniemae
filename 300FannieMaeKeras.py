# 
## Fannie Mae failed loan prediction
## Model to predict if a loan will have foreclosure costs
# keras requires python 2  / get tensorflow import error
# see forked project
# debugging pairplot 
# cast all variables to float - it choked on the decimal mixed with float
# limited the number of points in the query to 5000 - cuts way down on the plot time
from __future__ import print_function
!echo $PYTHON_PATH
import os, sys
#import path
from pyspark.sql import *

# create spark sql session
myspark = SparkSession\
    .builder\
    .appName("LoanPredictionSparkML_LogisticRegression") \
    .getOrCreate()



sc = myspark.sparkContext

import time
print ( time.time())

sc.setLogLevel("ERROR")
print ( myspark )
# make spark print text instead of octal
myspark.sql("SET spark.sql.parquet.binaryAsString=true")

# read in the data file from HDFS
# lots of sql cleanup, particularly removing NULL vals
# /user/hive/warehouse/fanniemae.db/fannie_harp_sql_normed_p
#
#
# use train dataset traindata_p
loansdf = myspark.read.parquet ( "/user/hive/warehouse/fanniemae.db/traindata_p")
# also read from s3 mydf = myspark.read.parquet ( "s3a://impalas3a/sample_07_s3a_parquet")
# print number of rows and type of object
loansdf.cache()
print ( loansdf.count() )
print  ( loansdf )
loansdf.show(5)
# create a table name to use for queries
loansdf.createOrReplaceTempView("loandata")
# run a query
# data is already normalized
# foreclosure costs range from a value of 0 to 1
lndf=myspark.sql('select * from loandata where fcl_costs < .9 limit 500000')
lndf.show(5)
lndf.count()

# pairplot to see what we have...
# not all cols are numeric, so we drop those for pair plot
# we will use string indexer later to clean those up

import seaborn as sns
import pandas

#debuging:
# convert all datatypes to float
# switch to python 3
# note: case statement in impala will create a decimal for
# case when a=b then 0.1   :-(

# reduce total columns in the pair plot so it doesn't take
# too long to run
pplotdf1 =myspark.sql('select seller_name, float(channel),float( intrate) ,float(loanamt),float(loan2val), float(numborrowers), float(creditscore),  float(fcl_costs) from loandata where fcl_costs<.8 limit 1000')
pplotdf1.show(3)
# seaborn wants a pandas dataframe, not a spark dataframe
# so convert
pdsdf = pplotdf1.toPandas()
pdsdf.head()

sns.set(style="ticks" , color_codes=True)
# this takes a long time to run:  
# you can see it if you uncomment it
g = sns.pairplot(pdsdf,  hue="seller_name" )

## Predict if a loan will have foreclosure costs


# expand to all columns for machine learning

# make the foreclosure costs binary 
# a loan has "failed" if foreclosure > 0
# in sql create column called "label" with 1 or 0

lndf = myspark.sql('select * from loandata ') #limit 500000')
# lndf = myspark.sql('select * from loandata limit 500000')
# 
# Impala did much of the normalization
# 
#
# show an example here 
# for those who want to use pyspark
# 
# need to convert from text field to numeric
# this is a common requirement when using sparkML
from pyspark.ml.feature import StringIndexer
# this will convert each unique string into a numeric
indexer = StringIndexer(inputCol="property_state", outputCol="loc_state")
indexed = indexer.fit(lndf).transform(lndf)
indexed.show(5)
## First try a logistic regression 
# now we need to create  a  "label" and "features"
# input for using the sparkML library

## This runs in the Cloudera Spark Cluster
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors

#
# the debt to income col has nulls
assembler = VectorAssembler(
    inputCols=[ "intrate", "loanamt", "loan2val", \
    "numborrowers",  \
               "creditscore", "loc_state"],
    outputCol="features")
      

# note the column headers - label and features are keywords
output = assembler.transform(indexed)
output.show(5)
output.count()

from pyspark.ml.classification import LogisticRegression


# Create a LogisticRegression instance. This instance is an Estimator.
lr = LogisticRegression(maxIter=10, regParam=0.3)
# Print out the parameters, documentation, and any default values.
print("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

# Learn a LogisticRegression model. This uses the parameters stored in lr.
model1 = lr.fit(output)

#### Major shortcut - no train and test data!!!
# Since model1 is a Model (i.e., a transformer produced by an Estimator),
# we can view the parameters it used during fit().
# This prints the parameter (name: value) pairs, where names are unique IDs for this
# LogisticRegression instance.
print("Model 1 was fit using parameters: ")
print(model1.extractParamMap())

trainingSummary = model1.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
# TPR true positive rate
# FPR false positive rate
trainingSummary.roc.show()
print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

prediction = model1.transform(output)
prediction.show(10)
result = prediction.select("label", "probability", "prediction") \
    .collect()
#print(result)

true0=0
false0=0
true1=0
false1=0


i=0
for row in result:
   if ( row.label == 0 and  row.prediction ==0 ):
      true0=true0+1
   if ( row.label == 0 and  row.prediction ==1 ):
      false1=false1+1
   if ( row.label == 1 and  row.prediction ==1 ):
      true1=true1+1
   if ( row.label == 1 and  row.prediction ==0 ):
      false0=false0+1
  
    #print("label=%s, prob=%s, prediction=%s" \
    #      % (row.label, row.probability, row.prediction))
    #comment: don't break the loop, get full error count if ( i > 10):
      #break
      
print ("true0=%i false0=%i true1=%i false1=%i"        % (true0 ,   false0 , true1 , false1 )) 

     

trainingSummary.roc.show()
print("areaUnderROC: " + str(trainingSummary.areaUnderROC)) 


## can we do better with a deep learning keras network?
#https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1

## This runs in the kubernetes-docker CDSW cluster
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# need dataframe for keras with only numerics
output.show(1)
kerasinputdf=output.drop('property_state','features','origination_date', 'loan_identifier','seller_name','fcl_costs')
kerasinputdf.show(1)
kerasinputdf.count()
kerasinputpsdf=kerasinputdf.toPandas()

kerasinputpsdf.head()
kerasinputpsdf.count()
kerasinputpsdf.describe(include='all')

sns.heatmap(kerasinputpsdf.corr(), annot=True)
# 
# we should not have "peeked" at the full dataset :-)
#

## traindataset at 80% of sample

# train and test split
trainpdf=kerasinputpsdf.sample(frac=0.8,random_state=200)
testpdf=kerasinputpsdf.drop(trainpdf.index)
trainpdf
testpdf


# creating input features and target variables
X= trainpdf.iloc[:,1:7]
y= trainpdf.iloc[:,0]
X.head()
y.head()



#from sklearn.cross_validation import train_test_split

#import sklearn
#from sklearn.model_selection import train_test_split
#from sklearn import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#!pip install --upgrade --force-reinstall tensorflow
#!pip install --upgrade --force-reinstall keras
import tensorflow as tf
import keras as ks
from keras import Sequential 
from keras.layers import Dense
from keras.callbacks import TensorBoard 
# tutorial from tensorflow.keras.callbacks import TensorBoard


classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal', input_dim=6))
#Second  Hidden Layer
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
# add another layer
#[[402254     99]
# [  8718     22]]
# to try to beat the above
classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal'))

#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

# tbcallback = tb(log_dir='/tmp/tblog', histogram_freq=0, write_graph=True, write_images=True)
#
# upgrade dask to fix? fork first to save working project
# !pip install --upgrade --force-reinstall dask
#tensorboard = TensorBoard(log_dir="/tmp/tblogs")
#
#
#Fitting the data to the training dataset
#classifier.fit(X,y, batch_size=10, epochs=50, verbose=1, callbacks=[tbcallback])
# the small batch size of 10 is hurting performance
# moving to 32
classifier.fit(X,y, batch_size=32, epochs=10, verbose=1)

classifier.summary()
#print(classifier.metrics()

# save the classifier to deploy
classifier.save("kerasLoanClassifier.keras.save")

eval_model=classifier.evaluate(X, y)
eval_model
# what do the input features look like
X.head
y_pred=classifier.predict(X)
y_pred =(y_pred>0.1)
# confusion matrix - barely correct when true
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)
print(cm)

def loanofficer(Z):
  return (classifier.predict(Z)>.65)

# function we will use for "production deployment"
loanofficer(X.head(2)) 

# need test data - have we just memorized the input data?

# try a multiple output final stage?
# more layers?  more cowbell?

## testing
# creating input features and target variables
X= testpdf.iloc[:,1:7]
y= testpdf.iloc[:,0]
X.head()
y.head()

eval_model=classifier.evaluate(X, y)
eval_model

y_pred=classifier.predict(X)
y_pred =(y_pred>0.07)
# confusion matrix - barely correct when true
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)
print(cm)

