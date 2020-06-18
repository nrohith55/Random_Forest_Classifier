# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:24:31 2020

@author: Rohith
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Random_Forest\\iris.csv")

train,test=train_test_split(df,test_size=0.3)

help(RandomForestClassifier)

model=RandomForestClassifier(n_estimators=100)
model.fit(train.iloc[:,0:4],train.iloc[:,4])

#To find train and test accuracy:

train_acc=np.mean(model.predict(train.iloc[:,0:4])==train.iloc[:,4])
test_acc=np.mean(model.predict(test.iloc[:,0:4])==test.iloc[:,4])

#######################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Random_Forest\\iris.csv")

X = df.iloc[:,0:4]
y = df.iloc[:,4]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

accuracy_score(y_test,y_pred)

confusion_matrix(y_test,y_pred)

classification_report(y_test,y_pred)




















