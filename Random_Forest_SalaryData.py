# -*- coding: utf-8 -*-
"""
Created on Sun May 17 00:50:01 2020

@author: Rohith
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Random_Forest\\SalaryData_Train.csv")
data=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Random_Forest\\SalaryData_Test.csv")

trainX=df.iloc[:,1:13]
trainy=df.iloc[:,13]
testX=data.iloc[:,1:13]
testY=data.iloc[:,0]

model=RandomForestClassifier(n_estimators=15)
model.fit(trainX,trainy)

string_columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']
from sklearn import preprocessing
for i in string_columns:
    number=preprocessing.LabelEncoder()
    trainX[i]=number.fit_transform(trainX[i])
    testX[i]=number.fit_transform(testX[i])
    
model.fit(trainX,trainy)

y_pred=model.predict(trainX)

from sklearn.metrics import confusion_matrix,accuracy_score

confusion_matrix(trainy,y_pred)

accuracy_score(trainy,y_pred)



###################################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Random_Forest\\SalaryData_Test.csv")

testX=data.iloc[:,1:13]
testY=data.iloc[:,0]

model=RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=15,criterion="entropy")
model.fit(testX,testY)

columns=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']
from sklearn import preprocessing
for i in columns:
    number=preprocessing.LabelEncoder()
    testX[i]=number.fit_transform(testX[i])
   
model.fit(testX,testY)

y_pred=model.predict(testX)

from sklearn.metrics import confusion_matrix,accuracy_score

confusion_matrix(testY,y_pred)
accuracy_score(testY,y_pred) #0.66
#################################################################################################################################



















