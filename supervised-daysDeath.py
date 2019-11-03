# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:15:07 2019

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

daysDeath =  result['days_to_death']
daysDeath = pd.DataFrame(daysDeath)



resultD = pd.merge(liverReact, daysDeath, on='patient_id')
resultD = resultD.dropna()

resultD['days_label'] = 0

for i in resultD.index:
    if resultD.at[i,'days_to_death'] > 700:
        resultD.at[i,'days_label'] = 1
#    if resultD.at[i,'days_to_death'] >= 1200:
#        resultD.at[i,'days_label'] = 2
#    

resultD = resultD.drop('days_to_death', axis = 1)

#SVM 

X = resultD.drop('days_label', axis=1)
y = resultD['days_label']

X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size = 0.20)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, Y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train,
                                                  test_size = .2,
                                                  random_state=12)
smD = SMOTE(random_state=1)
X_res, y_res = smD.fit_resample(x_train, y_train)


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_res, y_res)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))



#random forests

X_r = resultD.drop('days_to_death', axis=1)
y_r = resultD['days_to_death'].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_r, y_r, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#random forests for classification

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_r, y_r, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))











