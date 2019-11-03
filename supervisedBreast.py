# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 01:22:26 2019

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
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
from sklearn.ensemble import RandomForestClassifier


#read the dataframe with reactions and clinical data
result = pd.read_csv('resultB.csv')
result = result.set_index('patient_id')

#read df for metabolites
breast = pd.read_csv('dfB_.csv')
breast = breast.set_index('patient_id')

#reset the index
breast.reset_index(inplace=True)

#extract the id list
patient_id_B = breast['patient_id']

#change the id_names
final_idB = []
for i in patient_id_B:
    final_idB.append(i[:12])

xptoB = []
for i in final_idB:
    xptoB.append(i.replace('_', '-'))
    
breast['patient_id'] = xptoB
breast = breast.set_index('patient_id')


tumorStage = result['tumor_stage']

tumor = pd.DataFrame(tumorStage)

tumor['tumor_label'] = 0

for i in tumor.index:
    if tumor.at[i,'tumor_stage'] != 'stage i' and tumor.at[i,'tumor_stage'] != 'stage ia' and tumor.at[i,'tumor_stage'] != 'stage ib' and tumor.at[i,'tumor_stage'] != 'stage iia' and tumor.at[i,'tumor_stage'] != 'stage iib' and tumor.at[i,'tumor_stage'] != 'stage ii':
        tumor.at[i,'tumor_label'] = 1
        
breast['tumor_label'] = tumor['tumor_label'] 

#SVM 

X = breast.drop('tumor_label', axis=1)
y = breast['tumor_label']

X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size = 0.30)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, Y_train)

y_pred = svclassifier.predict(X_test)


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


#KNN

#KNN

X_knn = breast.drop('tumor_label', axis=1)
y_knn = breast['tumor_label'].values

#split dataset into train and test data
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.2, random_state=12, stratify=y_knn)

from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 2)
# Fit the classifier to the data
knn.fit(X_train_knn,y_train_knn)

#show predictions on the test data
y_predd = knn.predict(X_test_knn)

#check accuracy of our model on the test data
knn.score(X_test_knn, y_test_knn)
#0.6568627450980392

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test_knn,y_predd))
print(classification_report(y_test_knn,y_predd))
#print(accuracy_score(y_test, y_predd))

#split dataset into train and test data
x_train_knn, x_test_knn, Y_train_knn, Y_test_knn = train_test_split(X_train_knn, y_train_knn, test_size=0.2, stratify=y_train_knn)

#knn with smote

sm2 = SMOTE(random_state=12)
X_res_knn, y_res_knn = sm2.fit_resample(x_train_knn, Y_train_knn)

from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
# Fit the classifier to the data
knn.fit(X_res_knn,y_res_knn)

#show predictions on the test data
y_predd = knn.predict(X_test)

#check accuracy of our model on the test data
knn.score(X_test, y_test)
#0.6568627450980392

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_predd))
print(classification_report(y_test,y_predd))
print(accuracy_score(y_test, y_predd))

#knn with adasyn

sm3 = ADASYN(random_state=12)
X_res_knn, y_res_knn = sm3.fit_resample(x_train_knn, Y_train_knn)

from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 2)
# Fit the classifier to the data
knn.fit(X_res_knn,y_res_knn)

#show predictions on the test data
y_predd = knn.predict(X_test)

#check accuracy of our model on the test data
knn.score(X_test, y_test)
#0.6568627450980392

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_predd))
print(classification_report(y_test,y_predd))
print(accuracy_score(y_test, y_predd))



