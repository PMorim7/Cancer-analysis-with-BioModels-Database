# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:03:03 2019

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
for filename in glob.iglob('C:\\Users\\Pedro\\Desktop\\Thesis\\Biomodels-Liver\\**\*.xml',recursive=True):
    print(filename)
    reader = SBMLReader()
    document = reader.readSBML(filename)
    document.getNumErrors()
    m = document.getModel()
    id_list.append(m.id)
    
#create the dataframe
df = pd.DataFrame(0,index = id_list, columns = reactionsHMR200)
df.index.name = 'patient_id'

#transform HMR2.0 reactions on string
HMRreact = str(reactionsHMR200)


#if reactions exists on de model put 1
for filename in glob.iglob('C:\\Users\\Pedro\\Desktop\\Thesis\\Biomodels-Liver\\**\*.xml',recursive=True):
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
with open("react-liver.txt", "w") as file:
    file.write(str(id_list))
    
#save the DataFrame
df.to_csv('df-liver.csv')
df = pd.read_csv('df-liver.csv')
#remove duplicates on index
df = df.drop_duplicates()
df = df.set_index('patient_id')
#removing all columns with variance
def variance_threshold_selector(data, threshold):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]

#dataframe with variance 0.1
df_ = variance_threshold_selector(df, 0.05)
df_.to_csv('df-liver2.csv')
df_ = pd.read_csv('df-liver2.csv')

#clinical data
clinical = pd.read_csv('clinicalLiver.tsv', index_col = 1, sep = '\t')

#reset the index clinical
clinical.reset_index(inplace=True)

clinical = clinical.rename(index=str, columns={'submitter_id': 'patient_id'})

#read the liver df
liver = pd.read_csv('df-liver2.csv',sep =',',index_col=0)

#reset the index
liver.reset_index(inplace=True)

#extract the id list
patient_id = liver['patient_id']

#change the id_names
final_id = []
for i in patient_id:
    final_id.append(i[:12])

xpto = []
for i in final_id:
    xpto.append(i.replace('_', '-'))
    
liver['patient_id'] = xpto

#the two df, clinical and reactions
result = pd.merge(liver, clinical, on='patient_id')
result = result.set_index('patient_id')

result.to_csv('result.csv')
result = pd.read_csv('result.csv')
result = result.set_index('patient_id')
liver = liver.set_index('patient_id')


y = liver.index
model = SelectKBest(k=200)
model1 = model.fit_transform(liver,y)
model1.shape

# Create new dataframe with 200 columns
df__ = liver[liver.columns[model.get_support(indices=True)]]
df__

plt.figure()
plt.plot(dfCorr['HMR_9075'])
plt.plot(dfCorr['HMR_9074'])

#the best k for k-means -- Elbow method
X = pd.DataFrame(liver)

sse = {}
for k in range(1,10 ):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X)
    #X["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

#after this, the best k is 3

#k-means clustering
x = pd.DataFrame(liver)
#y1 = pd.DataFrame(y)


# K Means Cluster
model = KMeans(n_clusters=3)
model.fit(x)

# This is what KMeans thought
model.labels_

#K means Clustering 
def doKmeans(X, nclust=3):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = doKmeans(x, 3)
kmeans = pd.DataFrame(clust_labels)



#convert gender on 1 for male and 0 for female
#gender = result['gender']
#gen = []
#for i in gender:
#    if i == 'male':
#        gen.append('0')
#    else:
#        gen.append('1')
#        
#days = result['days_to_death']
#ss = liver
#ss['gender'] = gen
#ss['days_to_death'] = days
#ss.dropna()
#xp = pd.DataFrame(ss)


def doAgglomerative(x, nclust=3):
    model = AgglomerativeClustering(n_clusters=nclust, affinity = 'manhattan', linkage= 'complete')
    clust_labels1 = model.fit_predict(x)
    return (clust_labels1)

clust_labels1 = doAgglomerative(x, 3)
agglomerative = pd.DataFrame(clust_labels1)


##agglomerative clustering labels
#labelzeroA=[]
#labeloneA=[]
#labeltwoA=[]
#
#ll= len(clust_labels1)
#for i in range(0,ll):
#    if clust_labels1[i]==0:
#        labelzeroA.append(liver.index[i])
#    if clust_labels1[i]==1:
#        labeloneA.append(liver.index[i])
#    if clust_labels1[i]==2:
#        labeltwoA.append(liver.index[i])


#k-means clustering labels
labelzero=[]
labelone=[]
labeltwo=[]

ll= len(clust_labels)
for i in range(0,ll):
    if model.labels_[i]==0:
        labelzero.append(liver.index[i])
    if model.labels_[i]==1:
        labelone.append(liver.index[i])
    if model.labels_[i]==2:
        labeltwo.append(liver.index[i])

ll0 = len(labelzero)
ll1 = len(labelone)
ll2 = len(labeltwo)

countmale=0
countmale1=0
countmale2=0

for i in range(0,ll0):
    if (result.loc[labelzero[i],'gender'] == 'male'):     
        countmale = countmale+1

        
print(countmale,'ratio of males in the first set =',  countmale/ll0)    

for i in range(0,ll1):
    if (result.loc[labelone[i],'gender'] == 'male'):        
        countmale1 = countmale1+1
    
print(countmale1,'ratio of males in the second set =',  countmale1/ll1)  

for i in range(0,ll2):
    if (result.loc[labeltwo[i],'gender'] == 'male'):    
        countmale2 = countmale2+1

        
print(countmale2,'ratio of males in the third set =',  countmale2/ll2)   

#daystodeath k-means

rr0 =result.loc[labelzero,'days_to_death']
rrmean0=rr0.mean()

rr1 =result.loc[labelone,'days_to_death']
rrmean1=rr1.mean()

rr2 =result.loc[labeltwo,'days_to_death']
rrmean2=rr2.mean()


print('Mean 0 =', rrmean0, 'Mean 1 =',rrmean1, 'Mean 2=', rrmean2)


rr00 =result.loc[labelzero,'race']
#rrmean00=rr00.mean()

rr11 =result.loc[labeloneA,'race']
#rrmean11=rr11.mean()

rr22 =result.loc[labeltwoA,'race']
#rrmean11=rr11.mean()

rr000 =result.loc[labelzero,'vital_status']
Counter(rr000)

rr111 =result.loc[labelone,'vital_status']
Counter(rr111)
rr222 =result.loc[labeltwo,'vital_status']
Counter(rr222)

rrTk = result.loc[labelzero,'tumor_stage']
Counter(rrTk)
rrT1k = result.loc[labelone,'tumor_stage']
Counter(rrT1k)
rrT2k = result.loc[labeltwo,'tumor_stage']
Counter(rrT2k)

tumorStage = result.loc[labelzero,'tumor_stage']
tumorStage.value_counts().plot('pie')
plt.show()

tumorStage = result.loc[labelone,'tumor_stage']
tumorStage.value_counts().plot('pie')
plt.show()

tumorStage = result.loc[labeltwo,'tumor_stage']
tumorStage.value_counts().plot('pie')
plt.show()

l2 = liver.loc[labeltwo].sum()







ward = AgglomerativeClustering(n_clusters=3, linkage='ward');
ward.fit(liver);


labelzeroHL = []
labeloneHL = []
labeltwoHL = []

ll= len(clust_labels1)
for i in range(0,ll):
	if clust_labels1[i]==0:
		labelzeroHL.append(liver.index[i]);
	if clust_labels1[i]==1:
		labeloneHL.append(liver.index[i]);
	if clust_labels1[i]==2:
		labeltwoHL.append(liver.index[i]);

ll0HL= len(labelzeroHL)
ll1HL= len(labeloneHL)
ll2HL= len(labeltwoHL)

countmaleHL=0
countmaleHL1=0
countmaleHL2=0

for i in range(0,ll0HL):
	if (result.loc[labelzeroHL[i],'gender'] == 'male'):
		countmaleHL = countmaleHL+1;

print(countmaleHL,'ratio of males in the first set =',  countmaleHL/ll0HL);

for i in range(0,ll1HL):
	if (result.loc[labeloneHL[i],'gender'] == 'male'):
		countmaleHL1 = countmaleHL1+1;

print(countmaleHL1,'ratio of males in the second set =',  countmaleHL1/ll1HL);


for i in range(0,ll2HL):
	if (result.loc[labeltwoHL[i],'gender'] == 'male'):
		countmaleHL2 = countmaleHL2 + 1;

print(countmaleHL2,'ratio of males in the third set =',  countmaleHL2/ll2HL);

#daystodeath analysis agglomerative
rr0HL =result.loc[labelzeroHL,'days_to_death']
rrmean0HL=rr0HL.mean()

rr1HL =result.loc[labeloneHL,'days_to_death']
rrmean1HL=rr1HL.mean()

rr2HL =result.loc[labeltwoHL,'days_to_death']
rrmean2HL=rr2HL.mean()



print('Mean 0 =', rrmean0HL, 'Mean 1 =',rrmean1HL, 'Mean 2=', rrmean2HL)

rr0HLL =result.loc[labelzeroHL,'vital_status']
Counter(rr0HLL)

rr1HLL =result.loc[labeloneHL,'vital_status']
Counter(rr1HLL)
rr2HLL =result.loc[labeltwoHL,'vital_status']
Counter(rr2HLL)

rrT = result.loc[labelzeroHL,'tumor_stage']
Counter(rrT)
rrT1 = result.loc[labeloneHL,'tumor_stage']
Counter(rrT1)
rrT2 = result.loc[labeltwoHL,'tumor_stage']
Counter(rrT2)

#k-means vs agglomerative clustering groups
# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
bars1 = [153, 30, 155]
 
# Choose the height of the cyan bars
bars2 = [162, 26, 150]
  
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
plt.legend()
 
# Show graphic
plt.show()

#plot male/female
# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
bars1 = [102, 20, 107]
female = [51,10,48]
# Choose the height of the cyan bars
bars2 = [111, 17, 101]
female2 = [51,9,49]

# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
labels=['Label zero', 'Label one', 'Label two']
pos = np.arange(len(labels)) + 0.15

# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='Male - K-means clustering')
plt.bar(r1,female,width = barWidth,color='cyan',edgecolor='black',capsize=7,bottom=bars1, label='Female - K-means clustering')
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'orange', edgecolor = 'black', capsize=7, label='Male - Agglomerative Clustering')
plt.bar(r2,female2,width = barWidth,color='pink',edgecolor='black',capsize=7,bottom=bars2, label='Female - Agglomerative Clustering')
# general layout
plt.xticks(pos, labels)

plt.legend()
 
# Show graphic
plt.show()
#days to death analysis
# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
bars1 = [596.3, 982.4, 604.9]
 
# Choose the height of the cyan bars
bars2 = [587.7, 1095.7, 610.4]
  
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
plt.yticks(np.arange(0, 1600, 200))
plt.ylabel('Days to death')
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

#label 0 for k-means clustering
labels = ['Stage i', 'Stage ii', 'Stage iii', 'Stage iiia','Stage iiib','Stage iiic','Stage iv', 'not reported']
sizes = [76, 37, 1, 25,3,4,7]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','blue','pink','yellow','red']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()

#label 0 for agglomerative clustering
labels = ['Stage i', 'Stage ii', 'Stage iii', 'Stage iiia','Stage iiib','Stage iiic','Stage iv', 'not reported']
sizes = [82, 39,25,2,3,1,9]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral','blue','pink','yellow','red']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()


X = pd.DataFrame(liver)


from sklearn.manifold import MDS

mds = MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1, random_state=None, dissimilarity='euclidean')

newX = mds.fit_transform(X)



#lnear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

yR = result['days_to_death'].dropna()
yyR = pd.DataFrame(y)
linear = pd.merge(liver, yyR, on='patient_id')
linear = linear.drop(['days_to_death'], axis = 1)
XR = linear


X_train, X_test, y_train, y_test = train_test_split(XR, yR, test_size=0.4, random_state=101)

lm = LinearRegression()
lm.fit(X_train,y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.title('Linear regression - Reactions')

#pca 

from sklearn.preprocessing import StandardScaler

features = liver

x = StandardScaler().fit_transform(features)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


principalDf['patient_id'] = xpto
princi = principalDf.set_index('patient_id')

tumor = result['tumor_stage']

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


