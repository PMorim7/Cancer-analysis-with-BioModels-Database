# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:58:05 2019

@author: Pedro Morim
"""


import glob
import pandas as pd    
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import sklearn.metrics as sm
from libsbml import * 
from scipy.spatial.distance import cdist
from collections import Counter


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

#the two dataframes, clinical and reactions
resultB = pd.merge(dfB_, clinicalBreast, on='patient_id')
resultB = resultB.set_index('patient_id')

resultB.to_csv('resultB.csv')
resultB = pd.read_csv('resultB.csv',sep =',',index_col=0)
resultB = resultB.set_index('patient_id')

