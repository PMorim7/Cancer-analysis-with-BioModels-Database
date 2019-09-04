# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:58:05 2019

@author: Pedro
"""

#Breast Cancer

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

#read HMR2.0 Model
reader = SBMLReader()
document = reader.readSBML('HMRdatabase2_00.xml')
document.getNumErrors()
human = document.getModel()

#extract HMR2.0 recations of the model
reactionsHMR20 = []
for i in range(0,len(human.reactions)):
    reactionsHMR20.append(human.reactions[i].id)    
reactionsHMR20.sort()

reactionsHMR200=[]
for i in reactionsHMR20:
    reactionsHMR200.append(i[2:])

#size of HMR2.0 reactions 
size = len(reactionsHMR200)

#get id_list
id_list = []
for filename in glob.iglob('C:\\Users\\Pedro\\Desktop\\Thesis\\Biomodels\\**\*.xml',recursive=True):
    print(filename)
    reader = SBMLReader()
    document = reader.readSBML(filename)
    document.getNumErrors()
    m = document.getModel()
    id_list.append(m.id)


#create the dataframe
df = pd.DataFrame(0, index=id_list, columns=reactionsHMR200)
df.index.name = 'patient_id'

#transform HMR2.0 reactions on string
HMRreact = str(reactionsHMR200)

#if reactions exists on de model put 1
for filename in glob.iglob('C:\\Users\\Pedro\\Desktop\\Thesis\\Biomodels\\**\*.xml',recursive=True):
    print(filename)
    reader = SBMLReader()
    document = reader.readSBML(filename)
    document.getNumErrors()
    m = document.getModel()
    s = []
    s1 = []
    for i in range(0,len(m.reactions)):
        s = (str(m.reactions[i].id))
        s1 = (s[2:])
        index = HMRreact.find(s1)
        if index != -1:
            df.at[m.id,s1] = 1


#save the id_list of reactions
with open("react.txt", "w") as file:
    file.write(str(id_list))


#save the DataFrame
df.to_csv('df.csv')

#open df
dfB = pd.read_csv('df.csv')
dfB = dfB.set_index('patient_id')

#remove duplicates on index
dfB = dfB.drop_duplicates()

#removing all columns with variance
def variance_threshold_selector(data, threshold):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

#dataframe with variance 0.1
dfB_ = variance_threshold_selector(dfB, 0.05)

dfB_.to_csv('dfB_.csv')

### Clinical Data -- Breast ###

clinicalBreast = pd.read_csv('clinical.tsv', index_col = 1, sep = '\t')

#reset the index clinical
clinicalBreast.reset_index(inplace=True)

clinicalBreast = clinicalBreast.rename(index=str, columns={'submitter_id': 'patient_id'})

#read the breast df
dfB_ = pd.read_csv('dfB_.csv',sep =',',index_col=0)

#reset the index
dfB_.reset_index(inplace=True)

#extract the id list
patient_id_B = dfB_['patient_id']

#change the id_names
final_idB = []
for i in patient_id_B:
    final_idB.append(i[:12])

xptoB = []
for i in final_idB:
    xptoB.append(i.replace('_', '-'))
    
dfB_['patient_id'] = xptoB
breast = dfB_.set_index('patient_id')
#the two df, clinical and reactions
resultB = pd.merge(dfB_, clinicalBreast, on='patient_id')
resultB = resultB.set_index('patient_id')

resultB.to_csv('resultB.csv')
resultB = pd.read_csv('resultB.csv',sep =',',index_col=0)
resultB = resultB.set_index('patient_id')

#the best k -- Elbow method 

XB = pd.DataFrame(dfB_)

sse = {}
for k in range(1,10 ):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(XB)
    #XB["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("K")
plt.ylabel("SSE")
plt.title('The Elbow Method showing the optimal k')
plt.show()

## k means determine k
#distortions = []
#K = range(1,10)
#for k in K:
#    kmeanModel = KMeans(n_clusters=k).fit(XB)
#    kmeanModel.fit(XB)
#    distortions.append(sum(np.min(cdist(XB, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / XB.shape[0])
#
## Plot the elbow
#plt.plot(K, distortions, 'bx-')
#plt.xlabel('k')
#plt.ylabel('Distortion')
#plt.title('The Elbow Method showing the optimal k')
#plt.show()

# k = 3

dfB_.reset_index(inplace=True)
dfB_['patient_id'] = xptoB
dfBB = dfB_.set_index('patient_id')

#k-means clustering
x = pd.DataFrame(dfBB)


# K Means Cluster
model = KMeans(n_clusters=3)
model.fit(x)

# This is what KMeans thought
model.labels_

#size of groups 
Counter(model.labels_)

#data analysis

labelzero=[]
labelone=[]
labeltwo=[]

ll= len(model.labels_)
for i in range(0,ll):
    if model.labels_[i]==0:
        labelzero.append(dfBB.index[i])
    if model.labels_[i]==1:
        labelone.append(dfBB.index[i])
    if model.labels_[i]==2:
        labeltwo.append(dfBB.index[i])

ll0 = len(labelzero)
ll1 = len(labelone)
ll2 = len(labeltwo)

countmaleBA=0
countmaleBA1=0
countmaleBA2=0

#for the first set
for i in range(0,ll0):
    if (resultB.loc[labelzero[i],'gender'] == 'male'):
        countmaleBA = countmaleBA+1
       
print(countmaleBA,'ratio of males in the first set =',  countmaleBA/ll0)   

#for the second set

for i in range(0,ll1):
    if (resultB.loc[labelone[i],'gender'] == 'male'):
        countmaleBA1 = countmaleBA1+1
       
print(countmaleBA1,'ratio of males in the second set =',  countmaleBA1/ll0)  

#for the third set

for i in range(0,ll2):
    if (resultB.loc[labeltwo[i],'gender'] == 'male'):
        countmaleBA2 = countmaleBA2+1
       
print(countmaleBA2,'ratio of males in the third set =',  countmaleBA2/ll0)


top=0
count=0

rr0 =resultB.loc[labelzero,'days_to_death']
rrmean0=rr0.mean()

rr1 =resultB.loc[labelone,'days_to_death']
rrmean1=rr1.mean()


rr2 =resultB.loc[labeltwo,'days_to_death']
rrmean2=rr2.mean()

print('Mean 1 =', rrmean0, 'Mean 2 =',rrmean1, 'Mean 3 =', rrmean2)


tumorStage = resultB.loc[labelzero,'tumor_stage']
tumorStage.value_counts().plot('bar')
plt.show()

tumorStage = resultB.loc[labelone,'tumor_stage']
tumorStage.value_counts().plot('bar')
plt.show()

tumorStage = resultB.loc[labeltwo,'tumor_stage']
tumorStage.value_counts().plot('bar')
plt.show()

rr00 =resultB.loc[labelzero,'race']
#rrmean00=rr00.mean()
Counter(rr00)
rr11 =resultB.loc[labelone,'race']
#rrmean11=rr11.mean()
Counter(rr11)
rr22 =resultB.loc[labeltwo,'race']
#rrmean11=rr11.mean()
Counter(rr22)
## hierarchical clustering ##

rr00K =resultB.loc[labelzero,'vital_status']
#rrmean00=rr00.mean()
Counter(rr00K)

rr11K =resultB.loc[labelone,'vital_status']
#rrmean11=rr11.mean()
Counter(rr11K)

rr22K =resultB.loc[labeltwo,'vital_status']
#rrmean11=rr11.mean()
Counter(rr22K)

ward = AgglomerativeClustering(n_clusters=3, linkage='ward')
ward.fit(dfBB)

ward.labels_
Counter(ward.labels_)

#data analysis

labelzeroH=[]
labeloneH=[]
labeltwoH=[]

ll= len(ward.labels_)
for i in range(0,ll):
    if ward.labels_[i]==0:
        labelzeroH.append(dfBB.index[i])
    if ward.labels_[i]==1:
        labeloneH.append(dfBB.index[i])
    if ward.labels_[i]==2:
        labeltwoH.append(dfBB.index[i])

ll0H = len(labelzeroH)
ll1H = len(labeloneH)
ll2H = len(labeltwoH)
countmaleH=0

#for the first set
for i in range(0,ll0H):
    if (resultB.loc[labelzeroH[i],'gender'] == 'male'):
        countmaleH = countmaleH+1
       
print(countmaleH,'ratio of males in the first set =',  countmaleH/ll0H)   

#for the second set

for i in range(0,ll1H):
    if (resultB.loc[labeloneH[i],'gender'] == 'male'):
        countmaleH = countmaleH+1
       
print(countmaleH,'ratio of males in the second set =',  countmaleH/ll0)  

#for the third set

for i in range(0,ll2H):
    if (resultB.loc[labeltwoH[i],'gender'] == 'male'):
        countmaleH = countmaleH+1
       
print(countmaleH,'ratio of males in the third set =',  countmaleH/ll0)

top=0
count=0

rr0H =resultB.loc[labelzeroH,'days_to_death']
rrmean0H=rr0H.mean()

rr1H =resultB.loc[labeloneH,'days_to_death']
rrmean1H=rr1H.mean()


rr2H =resultB.loc[labeltwoH,'days_to_death']
rrmean2H=rr2H.mean()

print('Mean 1 =', rrmean0H, 'Mean 2 =',rrmean1H, 'Mean 3 =', rrmean2H)


rr00H =resultB.loc[labelzeroH,'race']
#rrmean00=rr00.mean()
Counter(rr00H)

rr11H =resultB.loc[labeloneH,'race']
#rrmean11=rr11.mean()
Counter(rr11H)

rr22H =resultB.loc[labeltwoH,'race']
#rrmean11=rr11.mean()
Counter(rr22H)

rr00HH =resultB.loc[labelzeroH,'vital_status']
#rrmean00=rr00.mean()
Counter(rr00HH)

rr11HH =resultB.loc[labeloneH,'vital_status']
#rrmean11=rr11.mean()
Counter(rr11HH)

rr22HH =resultB.loc[labeltwoH,'vital_status']
#rrmean11=rr11.mean()
Counter(rr22HH)

tumorStage = resultB.loc[labelzeroH,'tumor_stage']
tumorStage.value_counts().plot('bar')
plt.show()

tumorStage = resultB.loc[labeloneH,'tumor_stage']
tumorStage.value_counts().plot('bar')
plt.show()

tumorStage = resultB.loc[labeltwoH,'tumor_stage']
tumorStage.value_counts().plot('bar')
plt.show()



#plot k-means vs agglomerative clustering

# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
bars1 = [359, 171, 363]
 
# Choose the height of the cyan bars
bars2 = [412, 171, 310]
  
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
labels=['Label zero', 'Label one', 'Label two']
pos = np.arange(len(labels)) + 0.15
# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='K-means clustering')
 
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='Agglomerative Clustering')
 
# general layout
plt.xticks(pos, labels)
plt.yticks(np.arange(0, 600, 50))
plt.legend()
 
# Show graphic
plt.show()

#days to death analysis
# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
bars1 = [183.1, 228.2, 263.2]
 
# Choose the height of the cyan bars
bars2 = [230.63, 228.2, 213.7]
  
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
labels=['Label zero', 'Label one', 'Label two']
pos = np.arange(len(labels)) + 0.15
# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='K-means clustering')
 
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='Agglomerative Clustering')
 
# general layout
plt.xticks(pos, labels)
plt.yticks(np.arange(0, 400, 20))
plt.ylabel('Days to death')
plt.legend()
 
# Show graphic
plt.show()

#race analysis


# Create bars
#barWidth = 0.3
# Choose the height of the blue bars
bars1 = [254,116,262]
 
# Choose the height of the cyan bars
bars2 = [283,116,233]

bars3= [18,8,23]
bars4= [26,8,15]
  
bars5 = [52,30,59]
bars6 = [76,30,35]

bars7=[35,16,19]
bars8=[27,16,27]

bars9 = [0,1,0]
bars10 = [0,1,0]
 
# The X position of bars
r1 = [1,6,11]
r2 = [1+ barWidth,6,11]
r3 = [3,8,13]
r4 = [3+barWidth,8,13]
r5 = [5,10,15]
r6 = [5+barWidth,10,15] 
r7 = [7,12,17]
r8 = [7+barWidth,12,17]
r9 = [9,14,19]
r10 = [0+barWidth,14,19]

# Create barplot
plt.bar(r1, bars1, width = barWidth, color = (0.3,0.1,0.4,0.6), label='White - k-means clustering')
plt.bar(r2, bars2, width = barWidth, color = (0.3,0.5,0.4,0.6), label='White - Agglomerative clustering')
plt.bar(r3, bars3, width = barWidth, color = (0.3,0.9,0.4,0.6), label='Asian - K-means')
plt.bar(r4, bars4, width = barWidth, color = (0.4,0.9,0.6,0.8), label='Asian - Aggl')
plt.bar(r5, bars5, width = barWidth, color = ('brown'), label='Black or African American - k-means')
plt.bar(r6, bars6, width = barWidth, color = ('orange'), label='Black or African American - Aggl')
plt.bar(r7, bars7, width = barWidth, color = ('pink'), label='Not reported - k-means')
plt.bar(r8, bars8, width = barWidth, color = ('red'), label='Not reported - Aggl')
plt.bar(r9, bars9, width = barWidth, color = ('blue'), label='American indian - k-means')
plt.bar(r10, bars10, width = barWidth, color = ('yellow'), label='American indian - Aggl')

# Note: the barplot could be created easily. See the barplot section for other examples.
 
# Create legend
plt.legend()


# Adjust the margins
#plt.subplots_adjust(bottom= 0.2, top = 0.98)
 
# Show graphic
plt.show()

# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
bars1 = [18,8,23]
 
# Choose the height of the cyan bars
bars2 = [26,8,15]

# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
labels=['Label zero', 'Label one', 'Label two']
pos = np.arange(len(labels)) + 0.15
# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='K-means clustering')
 
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='Agglomerative Clustering')
 
# general layout
plt.xticks(pos, labels)
plt.yticks(np.arange(0, 400, 20))
plt.legend()
 
# Show graphic
plt.show()

# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
bars1 = [18,8,23]
 
# Choose the height of the cyan bars
bars2 = [26,8,15]
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
labels=['Label zero', 'Label one', 'Label two']
pos = np.arange(len(labels)) + 0.15
# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='K-means clustering')
 
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='Agglomerative Clustering')
 
# general layout
plt.xticks(pos, labels)
plt.yticks(np.arange(0, 40, 5))
plt.legend()

# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
bars1 = [52,30,59]

# Choose the height of the cyan bars
bars2 = [76,30,35]
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
labels=['Label zero', 'Label one', 'Label two']
pos = np.arange(len(labels)) + 0.15
# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='K-means clustering')
 
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='Agglomerative Clustering')
 
# general layout
plt.xticks(pos, labels)
plt.yticks(np.arange(0, 100, 10))
plt.legend()
 
# Show graphic
plt.show()
 
# Show graphic
plt.show()


# width of the bars
barWidth = 0.3

# Choose the height of the blue bars
bars1 = [35,16,19]

# Choose the height of the cyan bars
bars2 = [27,16,27]
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
labels=['Label zero', 'Label one', 'Label two']
pos = np.arange(len(labels)) + 0.15
# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='K-means clustering')
 
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='Agglomerative Clustering')
 
# general layout
plt.xticks(pos, labels)
plt.yticks(np.arange(0, 60, 10))
plt.legend()
 
# Show graphic
plt.show()
 



#plot alive/dead
# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
bars1 = [310, 151, 306]
female = [49,20,57]
# Choose the height of the cyan bars
bars2 = [355,151,261]
female2 = [57,20,49]

# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
labels=['Label zero', 'Label one', 'Label two']
pos = np.arange(len(labels)) + 0.15

# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='Alive - K-means clustering')
plt.bar(r1,female,width = barWidth,color='cyan',edgecolor='black',capsize=7,bottom=bars1, label='Dead - K-means clustering')
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'orange', edgecolor = 'black', capsize=7, label='Alive - Agglomerative Clustering')
plt.bar(r2,female2,width = barWidth,color='pink',edgecolor='black',capsize=7,bottom=bars2, label='Dead - Agglomerative Clustering')
# general layout
plt.xticks(pos, labels)
plt.yticks(np.arange(0, 800, 100))
plt.legend()
 
# Show graphic
plt.show()



#lnear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

yy = pd.DataFrame(y)
linear = pd.merge(breast, yy, on='patient_id')
lin = linear.drop(['days_to_death'], axis = 1)

X = lin
y = resultB['days_to_death'].dropna()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

lm = LinearRegression()
lm.fit(X_train,y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)



XB = pd.DataFrame(breast)

def doKmeans(X, nclust):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = doKmeans(XB, 2)
kmeans = pd.DataFrame(clust_labels)

#k-means clustering labels
labelzero=[]
labelone=[]

ll= len(clust_labels)
for i in range(0,ll):
    if clust_labels[i]==0:
        labelzero.append(breast.index[i])
    if clust_labels[i]==1:
        labelone.append(breast.index[i])

ll0 = len(labelzero) #ll0 = 37
ll1 = len(labelone) #ll1 = 301

#Days to death analysis
rr0 =resultB.loc[labelzero,'days_to_death']
rrmean0=rr0.mean()

rr1 =resultB.loc[labelone,'days_to_death']
rrmean1=rr1.mean()

vita = resultB.loc[labelzero,'vital_status']
vita1 = resultB.loc[labelone,'vital_status']


# Make a fake dataset:
height = [rrmean0, rrmean1]
bars = ('A', 'B')
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, height)
 
# Create names on the x-axis
plt.xticks(y_pos, bars)
 
# Show graphic
plt.show()


tumor = resultB.loc[labelzero,'tumor_stage']
tumor1 = resultB.loc[labelone,'tumor_stage']

tumor = resultB.loc[labelzero,'tumor_stage']
tumor.value_counts().plot('bar')
plt.show()

tumor1 = resultB.loc[labelone,'tumor_stage']
tumor1.value_counts().plot('bar')
plt.show()



resultB['label'] =0
resultB.loc[labelone,'label']=1

ss = sns.barplot(x='race', y='days_to_death', hue='label', data= resultB)
plt.show()


