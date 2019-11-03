# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:26:13 2019

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

#read the dataframe with reactions and clinical data
result = pd.read_csv('result.csv')
result = result.set_index('patient_id')

#read the dataframe with reactions
liverReact = pd.read_csv('df-liver22.csv')
liverReact = liverReact.set_index('patient_id')

#kmeans clustering

X = pd.DataFrame(liverReact)

def doKmeans(X, nclust):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = doKmeans(X, 2)
kmeans = pd.DataFrame(clust_labels)

#k-means clustering labels
labelzero=[]
labelone=[]

ll= len(clust_labels)
for i in range(0,ll):
    if clust_labels[i]==0:
        labelzero.append(liverReact.index[i])
    if clust_labels[i]==1:
        labelone.append(liverReact.index[i])

ll0 = len(labelzero) #ll0 = 37
ll1 = len(labelone) #ll1 = 301

#Days to death analysis
rr0 =result.loc[labelzero,'days_to_death']
rrmean0=rr0.mean()

rr1 =result.loc[labelone,'days_to_death']
rrmean1=rr1.mean()

sns.kdeplot(rr0, label='labelzero', shade=True)
sns.kdeplot(rr1, label='labelone', shade=True)

# rrmean0 = 1097.777
# rrmean1 = 584.911

l0 = liverReact.loc[labelzero].sum() # max(l0) = 37
l1 = liverReact.loc[labelone].sum() # max(l1) = 301

l0[((l0<200)&(l1>25))] # abaixo

l11R = l1[((l1<275)&(l0>28))]
l01R = pd.DataFrame(l11R)
l01R.index.name = 'React_id'
l01R.columns = ['Appears']
#HMR_1530    37
#HMR_1629    37
#HMR_1630    37
#HMR_3743    37
#HMR_3750    37
#HMR_3771    37
#HMR_3772    37
#HMR_3782    37
#HMR_3793    36
#HMR_3794    36
#HMR_3922    36
#HMR_4276    36
#HMR_5418    36
#HMR_5419    36
#HMR_6968    36

l00R = l0[((l0>302)&(l1<4))]
l0R = pd.DataFrame(l00R)
l0R.index.name = 'React_id'
l0R.columns = ['Appears']
#HMR_0339    297
#HMR_1870    296
#HMR_1872    297
#HMR_1896    296
#HMR_3799    299
#HMR_3965    298
#HMR_4187    299
#HMR_4342    299
#HMR_4516    298
#HMR_4641    300
#HMR_4708    301
#HMR_6566    301
#HMR_6623    300
#HMR_8489    299
#HMR_9359    299
#HMR_9360    299

#read df Metabolites and clinical data
resultMeta = pd.read_csv('resultMeta.csv')
resultMeta = resultMeta.set_index('patient_id')

#read df for metabolites
liverMeta = pd.read_csv('liverMeta.csv')
liverMeta = liverMeta.set_index('patient_id')

#kmeans clustering

# =============================================================================
def doAgglomerative(x, nclust=2):
    model = AgglomerativeClustering(n_clusters=nclust, affinity = 'manhattan', linkage= 'complete')
    clust_labels1 = model.fit_predict(x)
    return (clust_labels1)

clust_labels1 = doAgglomerative(liverMeta, 2)
agglomerative = pd.DataFrame(clust_labels1)




# ==========================================================XM = pd.DataFrame(liverMeta)
 
clust_labelsM, cent = doKmeans(XM, 2)
kmeansM = pd.DataFrame(clust_labelsM)
 
 #k-means clustering labels
labelzeroM=[]
labeloneM=[]
 
llM= len(clust_labels1)
for i in range(0,llM):
    if clust_labels1[i]==0:
        labelzeroM.append(liverMeta.index[i])
    if clust_labels1[i]==1:
        labeloneM.append(liverMeta.index[i])

ll0M = len(labelzeroM) #ll0 = 307
ll1M = len(labeloneM) #ll1 = 31===================

#Days to death metabolites analysis
rr0M =resultMeta.loc[labelzeroM,'days_to_death']
rrmean0M=rr0M.mean()

rr1M =resultMeta.loc[labeloneM,'days_to_death']
rrmean1M=rr1M.mean()

sns.kdeplot(rr0M, label='labelzero', shade=True)
sns.kdeplot(rr1M, label='labelone', shade=True)



# rrmean0M = 600.85
# rrmena1M = 982.42

l0M = liverMeta.loc[labelzeroM].sum() #31
l1M = liverMeta.loc[labeloneM].sum() #307

l000 = l0M[((l0M>29)&(l1M<280))]
l01 = pd.DataFrame(l000)
l01.index.name = 'Meta_id'
l01.columns = ['Appears']

l01.to_csv('MetaLiverA.csv')

MetaA = pd.read_excel('MetaLiverA.xlsx')
MetaA = pd.DataFrame(MetaA)
MetaA = MetaA.set_index('Meta_id')
MetaA.to_csv('MetaLiverA.csv')

l000 = l0M[((l1M<8)&(l0M>290))]
l00 = pd.DataFrame(l000)
l00.index.name = 'Meta_id'
l00.columns = ['Appears']

MetaB = pd.read_excel('metaLargeLiver.xlsx')
MetaB = pd.DataFrame(MetaB)
MetaB = MetaB.set_index('Meta_id')
MetaB.to_csv('MetaLiverB.csv')


s = set(labelzeroM)
s1 = set(labelzero)

sizeSet = s1.intersection(s)


#random groups for Reactions

L0 = liverReact.sample(n=38,replace = False)
L0 = L0.reset_index()

idL0=[]
for i in L0['patient_id']:
    idL0.append(i)

liverReact1= liverReact.reset_index()

idL1=[]
for i in liverReact1['patient_id']:
    idL1.append(i)

for i in idL1:
    for j in idL0:
        if i == j:
            idL1.remove(i)
            
#Days to death analysis
rr0Ran =result.loc[idL0,'days_to_death']
rrmean0Ran=rr0Ran.mean() #788.733

rr1Ran =result.loc[idL1,'days_to_death']
rrmean1Ran=rr1Ran.mean() #599.476

l0Ran = liverReact.loc[idL0].sum()
l1Ran = liverReact.loc[idL1].sum()

l00Ran = l0Ran[((l1Ran<275)&(l0Ran>36))]
#HMR_0803    37
#HMR_3743    37
#HMR_3750    37
#HMR_3771    37
#HMR_3772    37
#HMR_3782    37
#HMR_3794    37
#HMR_3922    37
#HMR_4283    37
#HMR_4685    37
#HMR_4687    37
#HMR_5418    37
#HMR_5419    37
#HMR_6749    37
#HMR_6754    37
#HMR_6761    37
#HMR_6767    37
#HMR_6791    37
#HMR_6968    37           

l11Ran = l0Ran[((l1Ran>302)&(l0Ran<4))
#HMR_0189    22
#HMR_0629    24
#HMR_3933    23
#HMR_4127    24
#HMR_4655    24
#HMR_4983    24
#HMR_5038    24
#HMR_5039    24
#HMR_5040    24
#HMR_5389    23
#HMR_8702    24


#labels for meta

L0M = liverMeta.sample(n=38,replace = False)
L0M = L0M.reset_index()

idL0M=[]
for i in L0M['patient_id']:
    idL0M.append(i)

liverMeta1= liverMeta.reset_index()

idL1M=[]
for i in liverMeta1['patient_id']:
    idL1M.append(i)

for i in idL1M:
    for j in idL0M:
        if i == j:
            idL1M.remove(i)
            
#Days to death analysis
rr0RanM =result.loc[idL0M,'days_to_death']
rrmean0RanM=rr0RanM.mean() #353.3125

rr1RanM =result.loc[idL1M,'days_to_death']
rrmean1RanM=rr1RanM.mean() #663.415

l0RanM = liverMeta.loc[idL0M].sum()
l1RanM = liverMeta.loc[idL1M].sum()

MetRan = l0RanM[((l1RanM<275)&(l0RanM>36))]
#m00198l    37
#m00572c    37
#m01116c    37
#m01434c    37
#m01959c    37
#m02015c    37
#m02481c    37
#m02525c    37
#m02870c    37
#m02910c    37

MetaRan1 = l0RanM[((l1RanM>302)&(l0RanM<4))]
#m00083m    29
#m00884m    29
#m01577m    29
#m01987c    29
#m02142m    29
#m02391m    29
#m02579s    29
#m02686c    25
#m02750r    28
#m02758c    25
#m03010m    29


y = liverMeta.index
model = SelectKBest(k=100)
model1 = model.fit_transform(liverMeta,y)
model1.shape

selec = liverMeta[liverMeta.columns[model.get_support(indices=True)]]

XSel = pd.DataFrame(selec)
 
clust_labelsSel, cent = doKmeans(XSel, 2)
kmeansSel = pd.DataFrame(clust_labelsSel)
 
 #k-means clustering labels
labelzeroSel=[]
labeloneSel=[]
 
llSel= len(clust_labelsSel)
for i in range(0,llSel):
    if clust_labelsSel[i]==0:
        labelzeroSel.append(selec.index[i])
    if clust_labelsSel[i]==1:
        labeloneSel.append(selec.index[i])

ll0Sel = len(labelzeroSel) #ll0 = 307
ll1Sel = len(labeloneSel) #ll1 = 31

resultSel = pd.merge(selec, resultMeta, on='patient_id')

rr0S =resultSel.loc[labelzeroSel,'days_to_death']
rrmean0S=rr0S.mean()

rr1S =resultSel.loc[labeloneSel,'days_to_death']
rrmean1S=rr1S.mean()

l0S = selec.loc[labelzeroSel].sum() #31
l1S = selec.loc[labeloneSel].sum() #307


l0S = l0S[((l0S>100)&(l1S<100))]