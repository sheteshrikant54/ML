import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model


a56_train_filepath = 'algebra_2005_2006_train_sample.txt'

a56data = pd.read_table(a56_train_filepath)
#a56data = pd.read_csv("trainValid.csv", header=0)
a56data.head()
hierarchy = a56data['Problem Hierarchy']
units, sections = [], []
for i in range(len(hierarchy)):
    units.append(hierarchy[i].split(',')[0].strip())
    sections.append(hierarchy[i].split(',')[1].strip())

# Now add 'Units' and 'Sections' as columns within the dataframe
a56data['Problem Unit'] = pd.Series(units, index=a56data.index)
a56data['Problem Section'] = pd.Series(sections, index=a56data.index)

# Rearrange order of columns
cols = a56data.columns.tolist()
cols = cols[0:3]+cols[-2::]+cols[3:-2]
a56data = a56data[cols]

df = a56data

students = set(a56data['Anon Student Id'])
print 'There are {0} students, so we will be adding as many columns to this dataframe.'.format(len(students))

numrows = len(df)

# Add a column for every student and fill them with zeros
for stud in students:
    df[stud] = pd.Series(np.zeros(numrows), index=df.index)
    
# For each student's problem entries in the dataframe, mark their column as 1
for stud in students:
    df.loc[df['Anon Student Id'] == stud,stud] = 1

print "Shape after adding student columns", np.shape(df)

units = set(a56data['Problem Unit'])
print 'There are {0} unique problem units, so we will be adding as many columns to this dataframe.'.format(len(units))

numrows = len(df)

# Add a column for every problem unit and fill them with zeros
for u in units:
    df[u] = pd.Series(np.zeros(numrows), index=df.index)
    
# For each student's attempt at a problem, mark the appropriate problem unit as 1
for u in units:
    df.loc[df['Problem Unit'] == u,u] = 1

print "Shape after adding Problem unit columns", np.shape(df)

sections = set(a56data['Problem Section'])
print 'There are {0} unique problem sections, so we will be adding as many columns to this dataframe.'.format(len(sections))

numrows = len(df)

# Add a column for every problem unit and fill them with zeros
for s in sections:
    df[s] = pd.Series(np.zeros(numrows), index=df.index)
    
# For each student's attempt at a problem, mark the appropriate problem unit as 1
for s in sections:
    df.loc[df['Problem Section'] == s,s] = 1

print "Shape after adding Problem Section columns", np.shape(df)


pnames = set(a56data['Problem Name'])
print 'There are {0} unique problem names, so we will be adding as many columns to this dataframe.'.format(len(pnames))

# Add a column for every problem unit and fill them with zeros
for n in pnames:
    df[n] = pd.Series(np.zeros(len(df)), index=df.index)
    
# For each student's attempt at a problem, mark the appropriate problem unit as 1
for n in pnames:
    df.loc[df['Problem Name'] == n,n] = 1

print "Shape after adding Problem Name columns", np.shape(df)


snames = set(a56data['Step Name'])
print 'There are {0} unique step names, so we will be adding as many columns to this dataframe.'.format(len(snames))

numrows = len(df)

# Add a column for every problem unit and fill them with zeros
for n in snames:
    df[n] = pd.Series(np.zeros(numrows), index=df.index)
    
# For each student's attempt at a problem, mark the appropriate problem unit as 1
for n in snames:
    df.loc[df['Step Name'] == n,n] = 1

print "Shape after adding Step Name columns", np.shape(df)

# Create an empty testing dataframe
testdf = pd.DataFrame(columns=df.columns)
unique_units = list(units)
# Create the testing set
for i in range(len(unique_units)):
    # Get the last problem of the current problem unit
    lastProb = list(df[df['Problem Unit'] == unique_units[i]]['Problem Name'])[-1]
    
    # Get all the rows corresponding to the last problem for the given problem unit
    lastProbRows = a56data[(df['Problem Unit'] == unique_units[i]) & (df['Problem Name']==lastProb)]
    
    # Concatenate test dataframe with the rows just found
    testdf = pd.concat([testdf,lastProbRows])

# Create a training dataframe that is equal to original dataframe with all the test cases removed
trainIndex = list(set(df.index) - set(testdf.index))
traindf = df.loc[trainIndex]

# Get the target feature within the test set: the Correct First Attmpt
CFAs = np.array(testdf['Correct First Attempt'])

# Define a helper function for calculating the root-mean-square error
def RMSE(p,y):
    ''' The Root-Mean-Square Error takes the predicted values p for the target
    variable y and takes the square root of the mean of the square of their
    differences. '''
    return np.sqrt(np.sum(np.square(p-y))/len(y))

# Test the RMSE for an array of all zeros
p = np.zeros(len(CFAs))
print 'An array of all zeros gives an RMSE of:',RMSE(p,CFAs)

# Test the RMSE for an array of all ones
p = np.ones(len(CFAs))
print 'An array of all ones gives an RMSE of:',RMSE(p,CFAs)

# Test the RMSE for an array of random 0s and 1s
p = np.random.randint(0,2,len(CFAs)).astype(float)
print 'An array of random ones and zeros gives an RMSE of:',RMSE(p,CFAs)

# Define the logistic function
def logisfunc(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit_plots(X,y,model,feat_names):
    num_samples  = np.shape(X)[0]
    num_features = np.shape(X)[1]
    print num_samples, 'number of samples'
    print num_features, 'number of features'
    
    # The coefficients and bias of the decision function
    coefs = model.coef_.ravel()
    bias  = model.intercept_
    
    # Plot the decision function separately for each feature
    x_plt = np.linspace(-1.0,1.0,300)
    
    fig = plt.figure(figsize=(3*num_features,6))
    
    for feat in range(num_features):
        x     = x_plt * coefs[feat] + bias
        decs  = logisfunc(x)
        
        fig.add_subplot(num_features,1,feat+1)
        plt.plot(X[:,feat],y,'x')
        plt.plot(x_plt,decs)
        plt.axis((0,1,0,1))
        plt.xlabel(feat_names[feat])
    plt.tight_layout()

def error_metrics(p,yy):
    '''Calculates the error metrics, i.e. the precision and recall.
    Precision = True positives / Predicted positives
    Recall    = True positives / Actual positives'''
    predicted_positives = len(p[p==1])
    actual_positives    = len(yy[yy==1])
    # The predicted values for when actual values are 1
    pp = p[yy==1]
    # True positives are when these predicted values are also 1
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

# Define a helper function to normalize the feature matrix X
import numba
def autonorm(X):
    ''' Calculates the mean and range of values of each column
    in the matrix (features) subtracts the mean from each value
    and divides by the range, thereby normalizing all values to
    fall between -1 and 1.'''
    x_means = np.mean(X,axis=0)
    x_means = np.ones(np.shape(X))*x_means
    x_maxs  = np.max(X,axis=0)
    x_mins  = np.min(X,axis=0)
    x_range = x_maxs - x_mins
    X_normd = (X - x_means) / x_range
    return X_normd

autonorm_jit = numba.jit(autonorm)


features_to_norm = ['Step Duration (sec)','Hints','Problem View']
binary_features = list(students)+list(units)+list(sections)+list(pnames)#+list(snames)
target_feature = ['Correct First Attempt']
all_features = features_to_norm + binary_features + target_feature

X = traindf[all_features].dropna()
y = np.array(X[target_feature]).astype(int).ravel()
X_to_norm = np.array(X[features_to_norm])
X_nonnorm = np.array(X[binary_features])
X_to_norm = autonorm(X_to_norm)
X = np.concatenate((X_to_norm,X_nonnorm), axis=1)

XX = testdf[all_features].dropna()
yy = np.array(XX[target_feature]).astype(int).ravel()
XX_to_norm = np.array(XX[features_to_norm])
XX_nonnorm = np.array(XX[binary_features])
XX_to_norm = autonorm(XX_to_norm)
XX = np.concatenate((XX_to_norm,XX_nonnorm), axis=1)


model = linear_model.LogisticRegression()

model.fit(X,y)
p = model.predict(XX).astype(float)

error_metrics(p,yy)
params = model.get_params(deep=True)

print params
T = model.predict_proba(X)

print T
