# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:40:21 2019

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
result = pd.read_csv('result.csv')
result = result.set_index('patient_id')

#read the dataframe with reactions
liverReact = pd.read_csv('df-liver22.csv')
liverReact = liverReact.set_index('patient_id')

tumorStage = result['tumor_stage']

tumor = pd.DataFrame(tumorStage)

tumor['tumor_label'] = 0

for i in tumor.index:
    if tumor.at[i,'tumor_stage'] != 'stage i':
        tumor.at[i,'tumor_label'] = 1
        
liverReact['tumor_label'] = tumor['tumor_label']

#SVM 

X = liverReact.drop('tumor_label', axis=1)
y = liverReact['tumor_label']

X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size = 0.20)


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

#SVM with ADASYN

ada = ADASYN(random_state=12, ratio={1:114, 0:102})
x_train_resA, y_train_resA = ada.fit_sample(x_train, y_train)

svclassifier = SVC(kernel='linear')
svclassifier.fit(x_train_resA, y_train_resA)

y_predA = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_predA))
print(classification_report(y_test,y_predA))
print(accuracy_score(y_test, y_predA))






#smote over sampling

def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
    

#KNN

X_knn = liverReact.drop('tumor_label', axis=1)
y_knn = liverReact['tumor_label'].values

#split dataset into train and test data
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y_knn, test_size=0.2, random_state=12, stratify=y_knn)

from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
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
print(accuracy_score(y_test, y_predd))

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

sm3 = ADASYN(random_state=1)
X_res_knn, y_res_knn = sm3.fit_resample(x_train_knn, Y_train_knn)

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


# =============================================================================
# from sklearn.model_selection import cross_val_score
# 
# #create a new KNN model
# knn_cv = KNeighborsClassifier(n_neighbors=3)
# #train model with cv of 5 
# cv_scores = cross_val_score(knn_cv, X_knn, y_knn, cv=5)
# #print each cv score (accuracy) and average them
# print(cv_scores)
# print('cv_scores mean:{}'.format(np.mean(cv_scores)))
# #[0.66666667 0.72058824 0.67164179 0.56716418 0.62686567]
# #cv_scores mean:0.6505853087503658
# 
# from sklearn.model_selection import GridSearchCV
# #create new a knn model
# knn2 = KNeighborsClassifier()
# #create a dictionary of all values we want to test for n_neighbors
# param_grid = {'n_neighbors': np.arange(1, 25)}
# #use gridsearch to test all values for n_neighbors
# knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
# #fit model to data
# knn_gscv.fit(X_knn, y_knn)
# 
# 
# #check top performing n_neighbors value
# knn_gscv.best_params_
# #n_neighbors: 15
# 
# #check mean score for the top performing value of n_neighbors
# knn_gscv.best_score_
# #0.7159763313609467
# =============================================================================

#neural networks

X_N = liverReact.drop('tumor_label', axis=1)
y_N = liverReact['tumor_label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_N, y_N)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(4116,4116))

mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))


sm3 = SMOTE(random_state=1)
X_res_N, y_res_N = sm3.fit_resample(X_N, y_N)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res_N, y_res_N)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)

StandardScaler(copy=True, with_mean=True, with_std=True)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(3108,3108),max_iter=250)

mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))

# =============================================================================
# [[50  7]
#  [12 49]]
#               precision    recall  f1-score   support
# 
#            0       0.81      0.88      0.84        57
#            1       0.88      0.80      0.84        61
# 
#     accuracy                           0.84       118
#    macro avg       0.84      0.84      0.84       118
# weighted avg       0.84      0.84      0.84       118
# =============================================================================


#random forests

X_R = liverReact.drop('tumor_label', axis=1)
y_R = liverReact['tumor_label']

X_train, X_test, y_train, y_test = train_test_split(X_R, y_R, test_size=0.2, random_state=0)

x_train, x_val, Y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

sm = SMOTE(random_state=12)
x_res, y_res = sm.fit_sample(x_train, Y_train)

regressorS = RandomForestClassifier(n_estimators=20, random_state=12)
regressorS.fit(x_res, y_res)
y_predS = regressorS.predict(X_test)

print(confusion_matrix(y_test,y_predS))
print(classification_report(y_test,y_predS))
print(accuracy_score(y_test, y_predS))



from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)
clf_rf.fit(x_res, y_res)

y_pred = svclassifier.predict(X_test)


# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

regressor = RandomForestClassifier(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))



#random forest

X_Ran = liverReact.drop('tumor_label', axis=1)
y_Ran = liverReact['tumor_label']



X_train, X_test, Y_train, y_test = train_test_split(X_Ran, y_Ran, test_size = 0.20)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=175)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,Y_train)

y_pred=clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

#random forest with smote

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=12)

sm1 = SMOTE(random_state=12)
x_res, y_res = sm1.fit_sample(x_train, y_train)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_res, y_res)

y_predS=clf.predict(X_test)

print(confusion_matrix(y_test,y_predS))
print(classification_report(y_test,y_predS))
print(accuracy_score(y_test, y_predS))

#random forest with adasyn

sm2 = ADASYN(random_state=12, ratio={1:114, 0:102})
x_res, y_res = sm2.fit_sample(x_train, y_train)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_res, y_res)

y_predS=clf.predict(X_test)

print(confusion_matrix(y_test,y_predS))
print(classification_report(y_test,y_predS))
print(accuracy_score(y_test, y_predS))



