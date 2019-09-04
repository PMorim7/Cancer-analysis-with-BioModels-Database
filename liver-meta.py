# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:40:00 2019

@author: Pedro
"""

# Liver Cancer

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
import cobra
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#read HMR2.0 Model 
reader = SBMLReader()
document = reader.readSBML('HMRdatabase2_00.xml')
document.getNumErrors()
human = document.getModel()

#extract HMR2.0 recations of the model
metaHMR20 = []
for i in range(0,len(human.species)):
    metaHMR20.append(human.species[i].id)    
metaHMR20.sort()

metaHMR200=[]
for i in metaHMR20:
    if i[0] == 'M':
        metaHMR200.append(i)
    
#size of HMR2.0 reactions 
size = len(metaHMR200)

#get id_list
id_list = []
for filename in glob.iglob('C:\\Users\\Pedro\\Desktop\\Thesis\\Biomodels - Liver\\**\*.xml',recursive=True):
    print(filename)
    reader = SBMLReader()
    document = reader.readSBML(filename)
    document.getNumErrors()
    m = document.getModel()
    id_list.append(m.id)

#create the dataframe
df = pd.DataFrame(0,index = id_list, columns = metaHMR200)
df.index.name = 'patient_id'

#transform HMR2.0 reactions on string
HMRmeta = str(metaHMR200)


#if reactions exists on de model put 1
for filename in glob.iglob('C:\\Users\\Pedro\\Desktop\\Thesis\\Biomodels - Liver\\**\*.xml',recursive=True):
    print(filename)
    reader = SBMLReader()
    document = reader.readSBML(filename)
    document.getNumErrors()
    m = document.getModel()
    s = []
    s1 = []
    for i in range(0,len(m.species)):
        s = (str(m.species[i].id))
        s1 = (s[2:])
        index = HMRmeta.find(s1)
        if index != -1:
            df.at[m.id,s1] = 1


#save the id_list of reactions
with open("react-liverMeta.txt", "w") as file:
    file.write(str(id_list))
    
#save the DataFrame
df.to_csv('df-liverMeta.csv')

df1 = pd.read_csv('df1-liverMeta.csv')
df1 = df1.set_index('patient_id')

def variance_threshold_selector(data, threshold):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

#dataframe with variance 0.1
df11 = variance_threshold_selector(df1, 0.000000001)
df11.to_csv('metaLiver.csv')

df1_ = pd.read_csv('df1-liverMeta2.csv')
df1_ = df1_.set_index('patient_id')

clinical = pd.read_csv('clinicalLiver.tsv', index_col = 1, sep = '\t')
clinical.reset_index(inplace=True)
clinical = clinical.rename(index=str, columns={'submitter_id': 'patient_id'})

resultMeta = pd.merge(df1_, clinical, on='patient_id')
resultMeta = resultMeta.set_index('patient_id')

resultMeta.to_csv('resultMeta.csv')


X = pd.DataFrame(df1_)

def doKmeans(X, nclust=3):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = doKmeans(X, 3)
kmeans = pd.DataFrame(clust_labels)


#k-means clustering labels
labelzero=[]
labelone=[]
labeltwo=[]

ll= len(clust_labels)
for i in range(0,ll):
    if clust_labels[i]==0:
        labelzero.append(df1_.index[i])
    if clust_labels[i]==1:
        labelone.append(df1_.index[i])
    if clust_labels[i]==2:
        labeltwo.append(df1_.index[i])

ll0 = len(labelzero)
ll1 = len(labelone)
ll2 = len(labeltwo)

countmale=0
countmale1=0
countmale2=0

for i in range(0,ll0):
    if (resultMeta.loc[labelzero[i],'gender'] == 'male'):     
        countmale = countmale+1

        
print(countmale,'ratio of males in the first set =',  countmale/ll0)    

for i in range(0,ll1):
    if (resultMeta.loc[labelone[i],'gender'] == 'male'):        
        countmale1 = countmale1+1
    
print(countmale1,'ratio of males in the second set =',  countmale1/ll1)  

for i in range(0,ll2):
    if (resultMeta.loc[labeltwo[i],'gender'] == 'male'):    
        countmale2 = countmale2+1

        
print(countmale2,'ratio of males in the third set =',  countmale2/ll2)   


rr0 =resultMeta.loc[labelzero,'days_to_death']
rrmean0=rr0.mean()

rr1 =resultMeta.loc[labelone,'days_to_death']
rrmean1=rr1.mean()

rr2 =resultMeta.loc[labeltwo,'days_to_death']
rrmean2=rr2.mean()

rrTk = resultMeta.loc[labelzero,'tumor_stage']
Counter(rrTk)
rrT1k = resultMeta.loc[labelone,'tumor_stage']
Counter(rrT1k)
rrT2k = resultMeta.loc[labeltwo,'tumor_stage']
Counter(rrT2k)

tumorStage = resultMeta.loc[labelzero,'tumor_stage']
tumorStage.value_counts().plot('pie')
plt.show()

tumorStage = resultMeta.loc[labelone,'tumor_stage']
tumorStage.value_counts().plot('pie')
plt.show()

tumorStage = resultMeta.loc[labeltwo,'tumor_stage']
tumorStage.value_counts().plot('pie')
plt.show()

rr000 =resultMeta.loc[labelzero,'vital_status']
Counter(rr000)

rr111 =resultMeta.loc[labelone,'vital_status']
Counter(rr111)
rr222 =resultMeta.loc[labeltwo,'vital_status']
Counter(rr222)


l0 = df1_.loc[labelzero].sum()
l1 = df1_.loc[labelone].sum()
l2 = df1_.loc[labeltwo].sum()


l0[((l1<100)&(l0>30))]

























#linear regression
y = resultMeta['days_to_death'].dropna()
yy = pd.DataFrame(y)

linear = pd.merge(df1_, yy, on='patient_id')
linear = linear.drop(['days_to_death'], axis = 1)

X = linear

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LinearRegression()
lm.fit(X_train,y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.title('Linear regression - Metabolites')

#data to 2 years
Days2years = resultMeta['days_to_death']
Days2years = Days2years.dropna()
days22 = pd.DataFrame(Days2years)

#linear regression 2 years
days22 = days22[days22.days_to_death <= 730]

react2years = pd.merge(df1_, days22, on='patient_id')
react2years = react2years.drop('days_to_death', axis = 1)

# Split the data into training/testing sets
x_train = react2years[:50]
x_test = react2years[51:]


# Split the targets into training/testing sets
y_train = days22[:50]
y_test = days22[51:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))


# Plot outputs
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


plt.scatter(y_test,y_pred)
plt.plot(y_test, y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())

plt.show()


#extract the id list
patient_id = df1_['patient_id']

#change the id_names
final_id = []
for i in patient_id:
    final_id.append(i[:12])

xpto = []
for i in final_id:
    xpto.append(i.replace('_', '-'))


from sklearn.preprocessing import StandardScaler

features = df1_

x = StandardScaler().fit_transform(features)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
df1_Prin = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

df1_Prin['patient_id'] = xpto
princi = df1_Prin.set_index('patient_id')

tumor = resultMeta['tumor_stage']

finalDf = pd.concat([princi, tumor], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['stage i', 'stage ii', 'stage iii', 'stage iiia','stage iiib', 'stage iiic', 'stage iv', 'not reported']
colors = ['red', 'grey', 'black', 'yellow', 'blue', 'c', 'brown']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['tumor_stage'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()




y = df1_.index
model = SelectKBest(k=1000)
model1 = model.fit_transform(df1_,y)
model1.shape
d200 = df1_[df1_.columns[model.get_support(indices=True)]]

#linear regression
yy = resultMeta['days_to_death'].dropna()
yyy = pd.DataFrame(yy)

linear2 = pd.merge(d200, yyy, on='patient_id')
linear2 = linear2.drop(['days_to_death'], axis = 1)

X = linear2

X_train, X_test, y_train, y_test = train_test_split(X, yy, test_size=0.4, random_state=0)

lm = LinearRegression()
lm.fit(X_train,y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.title('Linear regression - Metabolites')

y2 = yy.reset_index()
y2 = y2.drop('patient_id', axis = 1) 

#random forests

train_features, test_features, train_labels, test_labels = train_test_split(linear, y2, test_size=0.25, random_state=40)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)


# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels1)

test_labels1 = np.array(test_labels)


#rndom forest 2 years

train_features, test_features, train_labels, test_labels = train_test_split(react2years, days22, test_size=0.25)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000)


# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
test_labels1 = np.array(test_labels)
errors = abs(predictions - test_labels1)




# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels1)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')





# Labels are the values we want to predict
labels = np.array(days22['days_to_death'])
# Remove the labels from the features

# Saving feature names for later use
feature_list = list(react2years.columns)
# Convert to numpy array
features = np.array(react2years)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'days')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')