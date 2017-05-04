# import the necessary packages
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
import pandas as pd

train_filepath = 'algebra_2005_2006_train_160k.txt'

a56data = pd.read_table(train_filepath)
a56data.head()

#Diving Hierarchy column to units and sections
hierarchy = a56data['Problem Hierarchy']
units, sections = [], []
for i in range(len(hierarchy)):
    units.append(hierarchy[i].split(',')[0].strip())
    sections.append(hierarchy[i].split(',')[1].strip())

# Now add 'Units' and 'Sections' as columns within the dataframe
a56data['Problem Unit'] = pd.Series(units, index=a56data.index)
a56data['Problem Section'] = pd.Series(sections, index=a56data.index)

#Picking only useful columns
cols = ['Row', 'Anon Student Id', 'Problem Unit','Problem Section', 'Problem Name','Problem View', 'Step Name', 'Step Duration (sec)','Correct Step Duration (sec)', 'Error Step Duration (sec)','Correct First Attempt', 'Incorrects', 'Hints', 'Corrects']
df = a56data[cols]

sids = list(set(df['Anon Student Id']))
sid_dict = {}
for idx,sid in enumerate(sids):
    sid_dict[idx] = sid
    df.loc[df['Anon Student Id'] == sid,'Anon Student Id'] = idx

cat = 'Problem Unit'
prus = list(set(df[cat]))
pru_dict = {}
for idx,pru in enumerate(prus):
    pru_dict[idx] = pru
    df.loc[df[cat] == pru,cat] = idx
print 'Problem unit'

cat = 'Problem Section'
prss = list(set(df[cat]))
prs_dict = {}
for idx,prs in enumerate(prss):
    prs_dict[idx] = prs
    df.loc[df[cat] == prs,cat] = idx
print 'problem section'

cat = 'Problem Name'
prns = list(set(df[cat]))
prn_dict = {}
for idx,prn in enumerate(prns):
    prn_dict[idx] = prn
    df.loc[df[cat] == prn,cat] = idx
print 'problem name'

cat = 'Step Name'
stns = list(set(df[cat]))
stn_dict = {}
for idx,stn in enumerate(stns):
    stn_dict[idx] = stn
    df.loc[df[cat] == stn,cat] = idx
print 'step name'

df['Row'] = pd.Series(range(len(df)))

import numba
def autonorm(X):
    x_means = np.mean(X,axis=0)
    x_means = np.ones(np.shape(X))*x_means
    x_maxs  = np.max(X,axis=0)
    x_mins  = np.min(X,axis=0)
    x_range = x_maxs - x_mins
    X_normd = (X - x_means) / x_range
    return X_normd

autonorm_jit = numba.jit(autonorm)

df = df[cols[:8]+cols[10:]].dropna()
y = np.array(df['Correct First Attempt'])

testdf = pd.DataFrame(columns=df.columns)
#(trainX, testX, trainY, testY) = train_test_split(X_to_norm, y, test_size = 0.20)
unique_units = list(set(df['Problem Unit']))
for i in range(len(unique_units)):
    # Get the last problem of the current problem unit
    lastProb = list(df[df['Problem Unit'] == unique_units[i]]['Problem Name'])[-1]
    
    # Get all the rows corresponding to the last problem for the given problem unit
    lastProbRows = df[(df['Problem Unit'] == unique_units[i]) & (df['Problem Name']==lastProb)]
    
    # Concatenate test dataframe with the rows just found
    testdf = pd.concat([testdf,lastProbRows])

# Create a training dataframe that is equal to original dataframe with all the test cases removed
trainIndex = list(set(df.index) - set(testdf.index))
trainX = df.loc[trainIndex]
trainY = np.array(df['Correct First Attempt'].loc[trainIndex])
testX = df.loc[testdf.index]
testY = np.array(df['Correct First Attempt'].loc[testdf.index])

trainX = np.array(trainX[['Row', 'Anon Student Id', 'Problem Unit', 'Problem Section','Problem Name', 'Problem View', 'Step Name', 'Step Duration (sec)', 'Incorrects', 'Hints', 'Corrects']])
trainX = autonorm(trainX)

testX = np.array(testX[['Row', 'Anon Student Id', 'Problem Unit', 'Problem Section','Problem Name', 'Problem View', 'Step Name', 'Step Duration (sec)', 'Incorrects', 'Hints', 'Corrects']])
testX = autonorm(testX)


from sklearn import svm
clf = svm.SVC()
clf.fit(trainX, trainY.astype(int))

predX = clf.predict(testX)
print classification_report(testY, predX)
