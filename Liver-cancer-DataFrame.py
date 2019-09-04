# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:03:03 2019

@author: Pedro Morim 
"""

#all packages that were used
import glob
import pandas as pd    
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import sklearn.metrics as sm
from libsbml import * 
from scipy.spatial.distance import cdist
from collections import Counter


#First, we downloaded the HMR2.0 model from "http://www.metabolicatlas.org"
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

#Next, each patient's model was downloaded from BioModels Database
#get the id_list each patient
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

#if reactions exists on the model put 1
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

df_ = variance_threshold_selector(df, 0.05)
df_.to_csv('df-liver2.csv')
df_ = pd.read_csv('df-liver2.csv')

#The clinical data was downloaded from the https://portal.gdc.cancer.gov/
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

#join the two dataframes, clinical data and reactions
result = pd.merge(liver, clinical, on='patient_id')
result = result.set_index('patient_id')

result.to_csv('result.csv')
result = pd.read_csv('result.csv')
result = result.set_index('patient_id')
liver = liver.set_index('patient_id')


