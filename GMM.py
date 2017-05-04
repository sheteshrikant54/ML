import numpy as np
import pandas as pd
from sklearn import mixtures
from sklearn import preprocessing, datasets
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import csv
import time



start_time = time.time()

df = pd.read_csv("trainValid.csv", header=0)

cfa = df['Correct First Attempt'].ravel(order='C')

studentId = df['Anon Student Id'].ravel(order='C')
stepName = df['Step Name'].ravel(order='C')


le_studentId = preprocessing.LabelEncoder()
le_stepName = preprocessing.LabelEncoder()


encoded_studentId = le_studentId.fit_transform(studentId)
encoded_stepName = le_stepName.fit_transform(stepName)



studentId_dict = dict(zip(studentId, encoded_studentId))
stepName_dict = dict(zip(stepName, encoded_stepName))



x_array = []

for i in range(len(encoded_studentId)):
    x_array.append([encoded_studentId[i], encoded_stepName[i]])

# Convert x_array to np matrix
X = np.matrix(x_array)

# set Y (labels) to be our cfa
Y = cfa





k = 101
GMM = mixture.GaussianMixture(n_components=k, tol=0.01, init_params='kmeans')
GMM.fit(X, Y)


# **************************************************************************

def predict_single_cfa(studentId, stepName):
    encoded_student_id = studentId_dict.get(studentId)
    print("encoded_student_id: ", encoded_student_id)
    encoded_step_name = stepName_dict.get(stepName)
    print("encoded_step_name: ", encoded_step_name)
    prediction = GMM.predict([[encoded_student_id, encoded_step_name]])
    print("prediction: ", prediction[0])






def predict(encoded_studentId, encoded_stepName):

    prediction_array = np.array([])

    for row in range(len(encoded_studentId)):
        prediction = GMM.predict_proba([[encoded_studentId[row], encoded_stepName[row]]])
        prediction_array = np.append(prediction_array, prediction[0, 1])
        print("Prediction for row {:d} = {:f}".format(row, prediction[0, 1]))

    print("prediction_array: ", prediction_array)

    return prediction_array


# **************************************************************************

#  calculate RMSE:

def calculate_rmse(prediction_array, ground_truth_array):
    rmse = math.sqrt(mean_squared_error(ground_truth_array, prediction_array))
    print("RMSE ", rmse)

# **************************************************************************

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
    
    #print 'Root-mean-square error: ', RMSE(p,yy)
    
    print '\nPrecision: Of all predicted CFAs, what fraction actually succeeded?'
    print precision
    
    print '\nRecall: Of all actual CFAs, what fraction did we predict correctly?'
    print recall
    
    print '\nF_1 Score: ', F_1score


def main():

    # testFileName = input("Please enter your test file name: ")
    testFileName = "testValid.csv"
    testFile = pd.read_csv(testFileName, header=0)
    #df = pd.read_csv("train.csv", header=0)
    #train, testFile = train_test_split(df, test_size = 0.01)
    print("Loading... :) ")

    # CFA is our ground_truth
    ground_truth_array = testFile['Correct First Attempt'].dropna().ravel(order='C')

    # Student ID and Step ID will be our X   (input for prediction)
    test_studentId = testFile['Anon Student Id'].ravel(order='C')
    test_stepName = testFile['Step Name'].ravel(order='C')
    test_row = testFile['Row'].ravel(order='C')
    # Generate label encoders
    le_test_studentId = preprocessing.LabelEncoder()
    le_test_stepName = preprocessing.LabelEncoder()

    # Encode student id and step name values
    encoded_test_studentId = le_test_studentId.fit_transform(test_studentId)
    encoded_test_stepName = le_test_stepName.fit_transform(test_stepName)

    prediction_array = predict(encoded_test_studentId, encoded_test_stepName)

    calculate_rmse(prediction_array, ground_truth_array)
    error_metrics(prediction_array, ground_truth_array)

    # Save results in a CSV:
    '''
    with open('GMM_2_features_k723.csv', 'w') as csvfile:
        fieldnames = ['Row', 'Student ID', 'Correct First Attempt', 'Ground Truth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for x in range(len(test_studentId)):
            writer.writerow(
                {'Row': test_row[x], 'Student ID': test_studentId[x], 'Correct First Attempt': prediction_array[x],
                'Ground Truth': ground_truth_array[x]})
    '''

if __name__ == '__main__':
    main()

print("K = ", k)
print("GMM Classifier")
print("--- Total time: %s seconds ---" % (time.time() - start_time))





