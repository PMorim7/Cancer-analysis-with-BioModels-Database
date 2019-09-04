# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:40:00 2019

@author: Pedro Morim
"""

# Liver Cancer

import glob
import pandas as pd    
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import sklearn.metrics as sm
from libsbml import * 
from collections import Counter
import cobra


#read HMR2.0 Model 
reader = SBMLReader()
document = reader.readSBML('HMRdatabase2_00.xml')
document.getNumErrors()
human = document.getModel()

#extract HMR2.0 metabolites of the model
metaHMR20 = []
for i in range(0,len(human.species)):
    metaHMR20.append(human.species[i].id)    
metaHMR20.sort()

metaHMR200=[]
for i in metaHMR20:
    if i[0] == 'M':
        metaHMR200.append(i)
    
#size of HMR2.0 metabolites 
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


