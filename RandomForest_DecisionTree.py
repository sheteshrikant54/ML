import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cross_validation import cross_val_score
import numba
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

a56_train_filepath =  'algebra_2005_2006_train_160k.txt'
a56data = pd.read_table(a56_train_filepath)
print a56data.head()

hierarchy = a56data['Problem Hierarchy']
units, sections = [], []
for i in range(len(hierarchy)):
    units.append(hierarchy[i].split(',')[0].strip())
    sections.append(hierarchy[i].split(',')[1].strip())
print 'created units and section'

a56data['Problem Unit'] = pd.Series(units, index=a56data.index)
a56data['Problem Section'] = pd.Series(sections, index=a56data.index)

cols = a56data.columns.tolist()
cols = cols[0:3]+cols[-2::]+cols[3:-2]
a56data = a56data[cols]
df = a56data
cats = ['Anon Student Id', 'Problem Hierarchy', 'Problem Unit', 'Problem Section', 'Problem Name']

sids = list(set(df['Anon Student Id']))
sid_dict = {}
for idx,sid in enumerate(sids):
    sid_dict[idx] = sid
    df.loc[df['Anon Student Id'] == sid,'Anon Student Id'] = idx
print 'updated student id'

cat = 'Problem Hierarchy'
prhs = list(set(df[cat]))
prh_dict = {}
for idx,prh in enumerate(prhs):
    prh_dict[idx] = prh
    df.loc[df[cat] == prh,cat] = idx
print 'updated problem  hierarchy'

cat = 'Problem Unit'
prus = list(set(df[cat]))
pru_dict = {}
for idx,pru in enumerate(prus):
    pru_dict[idx] = pru
    df.loc[df[cat] == pru,cat] = idx
print 'updated problem units'
cat = 'Problem Section'
prss = list(set(df[cat]))
prs_dict = {}
for idx,prs in enumerate(prss):
    prs_dict[idx] = prs
    df.loc[df[cat] == prs,cat] = idx
print 'updated problem section'
cat = 'Problem Name'
prns = list(set(df[cat]))
prn_dict = {}
for idx,prn in enumerate(prns):
    prn_dict[idx] = prn
    df.loc[df[cat] == prn,cat] = idx
print 'updated problem name'
cat = 'Step Name'
stns = list(set(df[cat]))
stn_dict = {}
for idx,stn in enumerate(stns):
    stn_dict[idx] = stn
    df.loc[df[cat] == stn,cat] = idx
print 'updated step nm'

testdf = pd.DataFrame(columns=df.columns)

unique_units = list(set(df['Problem Unit']))
for i in range(len(unique_units)):
    lastProb = list(df[df['Problem Unit'] == unique_units[i]]['Problem Name'])[-1]
    lastProbRows = a56data[(df['Problem Unit'] == unique_units[i]) & (df['Problem Name']==lastProb)]
    testdf = pd.concat([testdf,lastProbRows])

trainIndex = list(set(df.index) - set(testdf.index))
traindf = df.loc[trainIndex]
print 'test data created'

CFAs = np.array(testdf['Correct First Attempt'])

def RMSE(p,y):
    ''' The Root-Mean-Square Error takes the predicted values p for the target
    variable y and takes the square root of the mean of the square of their
    differences. '''
    return np.sqrt(np.sum(np.square(p-y))/len(y))

p = np.zeros(len(CFAs))
print 'An array of all zeros gives an RMSE of:',RMSE(p,CFAs)

p = np.ones(len(CFAs))
print 'An array of all ones gives an RMSE of:',RMSE(p,CFAs)

p = np.random.randint(0,2,len(CFAs)).astype(float)
print 'An array of random ones and zeros gives an RMSE of:',RMSE(p,CFAs)

def error_metrics(p,yy):
    predicted_positives = len(p[p==1])
    actual_positives    = len(yy[yy==1])

    pp = p[yy==1]
    true_positives      = len(pp[pp==1])
    false_positives     = len(yy) - true_positives
    
    precision = float(true_positives) / float(predicted_positives)
    recall    = float(true_positives) / float(actual_positives)
    
    F_1score  = 2.0 * precision * recall / (precision + recall)
    
    print 'Root-mean-square error: ', RMSE(p,yy)
    
    print '\nPrecision: Of all predicted CFAs, what fraction actually succeeded?'
    print precision
    
    print '\nRecall: Of all actual CFAs, what fraction did we predict correctly?'
    print recall
    
    print '\nF_1 Score: ', F_1score

def autonorm(X):
    x_means = np.mean(X,axis=0)
    x_means = np.ones(np.shape(X))*x_means
    x_maxs  = np.max(X,axis=0)
    x_mins  = np.min(X,axis=0)
    x_range = x_maxs - x_mins
    X_normd = (X - x_means) / x_range
    return X_normd

autonorm_jit = numba.jit(autonorm)

features_to_norm = ['Step Duration (sec)','Hints','Problem View']
category_features = ['Anon Student Id', 'Problem Hierarchy', 'Problem Unit', 'Problem Section', 'Problem Name']
target_feature = ['Correct First Attempt']
features = features_to_norm + category_features
all_features = features_to_norm + category_features + target_feature

X = traindf[all_features].dropna()
y = np.array(X[target_feature]).astype(int).ravel()
X_to_norm = np.array(X[features_to_norm])
X_nonnorm = np.array(X[category_features])
X_to_norm = autonorm(X_to_norm)
X = np.concatenate((X_to_norm,X_nonnorm), axis=1)

XX = testdf[all_features].dropna()
yy = np.array(XX[target_feature]).astype(int).ravel()
XX_to_norm = np.array(XX[features_to_norm])
XX_nonnorm = np.array(XX[category_features])
XX_to_norm = autonorm(XX_to_norm)
XX = np.concatenate((XX_to_norm,XX_nonnorm), axis=1)

print 'train data created'
model = tree.DecisionTreeClassifier()

model = model.fit(X,y)

p = model.predict(XX).astype(float)
error_metrics(p,yy)
scores = cross_val_score(model, X, y)
print 'Accuracy: {0:5.2f} (+/-{1:5.2f})'.format(scores.mean(), scores.std()*2)

model = RandomForestClassifier(n_estimators=70, criterion="entropy", max_features=8)
model = model.fit(X,y)
p = model.predict(XX).astype(float)

error_metrics(p,yy)
importances = model.feature_importances_
n_feats = len(features)
feat_std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

print("\nFeature ranking:")

for f in range(n_feats):
    print '{0:2} - {1:20}: {2:5.4f} (std: {3:5.4f})'.format(f+1,features[indices[f]],importances[indices[f]],feat_std[indices[f]])

scores = cross_val_score(model, X, y)
print 'Accuracy: {0:5.2f} (+/-{1:5.2f})'.format(scores.mean(), scores.std()*2)

