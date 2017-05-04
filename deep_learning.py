
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
import pandas as pd

train_filepath = '/home/mahesh/Dropbox/MS Sem 2/ML/Self-selected/Data/algebra_2005_2006/algebra_2005_2006_train_160k.txt'

a56data = pd.read_table(train_filepath)
a56data.head()

hierarchy = a56data['Problem Hierarchy']
units, sections = [], []
for i in range(len(hierarchy)):
    units.append(hierarchy[i].split(',')[0].strip())
    sections.append(hierarchy[i].split(',')[1].strip())

a56data['Problem Unit'] = pd.Series(units, index=a56data.index)
a56data['Problem Section'] = pd.Series(sections, index=a56data.index)

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

cat = 'Problem Section'
prss = list(set(df[cat]))
prs_dict = {}
for idx,prs in enumerate(prss):
    prs_dict[idx] = prs
    df.loc[df[cat] == prs,cat] = idx

cat = 'Problem Name'
prns = list(set(df[cat]))
prn_dict = {}
for idx,prn in enumerate(prns):
    prn_dict[idx] = prn
    df.loc[df[cat] == prn,cat] = idx

cat = 'Step Name'
stns = list(set(df[cat]))
stn_dict = {}
for idx,stn in enumerate(stns):
    stn_dict[idx] = stn
    df.loc[df[cat] == stn,cat] = idx

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


X_to_norm = np.array(df[['Row', 'Anon Student Id', 'Problem Unit', 'Problem Section','Problem Name', 'Problem View', 'Step Name', 'Step Duration (sec)', 'Incorrects', 'Hints', 'Corrects']])
X_to_norm = autonorm(X_to_norm)

(trainX, testX, trainY, testY) = train_test_split(X_to_norm, y, test_size = 0.20)

dbn = DBN([trainX.shape[1], 300, 2],learn_rates = 0.3, learn_rate_decays = 0.9,epochs = 10,verbose = 1)

dbn.fit(trainX.astype(int), trainY.astype(int))
pred_deepX = dbn.predict(testX.astype(int))
print classification_report(testY, pred_deepX)
