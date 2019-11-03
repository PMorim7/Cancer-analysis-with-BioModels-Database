# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:54:03 2019

@author: Pedro
"""

import glob
import pandas as pd    
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from libsbml import * 
from scipy.spatial.distance import cdist
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#read the dataframe with reactions and clinical data
result = pd.read_csv('result.csv')
result = result.set_index('patient_id')

#read the dataframe with reactions
liverReact = pd.read_csv('df-liver22.csv')
liverReact = liverReact.set_index('patient_id')

vital = result['vital_status']

vital = pd.DataFrame(vital)

vital['vital_label'] = 0
for i in vital.index:
    if vital.at[i,'vital_status'] == 'dead':
        vital.at[i,'vital_label'] = 1
        
liverReact['vital_label'] = vital['vital_label']

#SVM 

X = liverReact.drop('vital_label', axis=1)
y = liverReact['vital_label']

X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size = 0.30)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


#SVM with SMOTE

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train,
                                                  test_size = .2,
                                                  random_state=12)
smote = SMOTE(random_state=12)
x_train_res, y_train_res = smote.fit_sample(x_train, y_train)

svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train_res, y_train_res)

y_predS = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_predS))
print(classification_report(y_test,y_predS))
print(accuracy_score(y_test, y_predS))


