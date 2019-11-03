# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:41:43 2019

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

#read df for metabolites
liverMeta = pd.read_csv('liverMeta.csv')
liverMeta = liverMeta.set_index('patient_id')


tumorStage = result['tumor_stage']

tumor = pd.DataFrame(tumorStage)

tumor['tumor_label'] = 0

for i in tumor.index:
    if tumor.at[i,'tumor_stage'] != 'stage i' and tumor.at[i,'tumor_stage'] != 'stage ii':
        tumor.at[i,'tumor_label'] = 1
        
liverMeta['tumor_label'] = tumor['tumor_label']

sns.pairplot(liverMeta, hue='tumor_label', size=2.5)

#SVM 

X = liverMeta.drop('tumor_label', axis=1)
y = liverMeta['tumor_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


#KNN

X_knn = liverMeta.drop('tumor_label', axis=1)
y_knn = liverMeta['tumor_label'].values

#split dataset into train and test data
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.4, random_state=1, stratify=y_knn)

from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 7, metric = 'kulsinski')
# Fit the classifier to the data
knn.fit(X_train_knn,y_train_knn)

#show predictions on the test data
y_predd=knn.predict(X_test_knn)

#check accuracy of our model on the test data
knn.score(X_test_knn, y_test_knn)

print(confusion_matrix(y_test_knn,y_predd))
print(classification_report(y_test_knn,y_predd))


from sklearn.model_selection import cross_val_score

#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X_knn, y_knn, cv=5)
#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))


from sklearn.model_selection import GridSearchCV
#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X_knn, y_knn)


#check top performing n_neighbors value
knn_gscv.best_params_
#n_neighbors: 6

#check mean score for the top performing value of n_neighbors
knn_gscv.best_score_
#0.7041420118343196


#neural networks

X_N = liverMeta.drop('tumor_label', axis=1)
y_N = liverMeta['tumor_label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_N, y_N)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(3108,3108),max_iter=250)

mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))


















