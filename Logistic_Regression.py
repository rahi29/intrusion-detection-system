import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

################################################################### Read data ##########################################################################

data = pd.read_csv('./Data/filtered_data.csv')


##################################################### Convert 'class' column to binary numeric values ####################################################


data['class'] = (data['class'] != "normal").astype(int)


###################################################### ##########Split features and target ################################################################


X = data.drop(columns=['class'])
y = data['class']

#############################################################Define the train-test split ratio ############################################################


train_size = 0.9
train_samples = int(len(X) * train_size)


##############################################################Split data into train and test sets ##########################################################


X_train = X[:train_samples]
y_train = y[:train_samples]
X_test = X[train_samples:]
y_test = y[train_samples:]

############################################################# Initialize and fit Logistic Regression model ##################################################


clf = LogisticRegression()
clf.fit(X_train, y_train)

######################################################################## Predict on test data #################################################################


pred_target = clf.predict(X_test)



########################################################################## Calculate accuracy ##################################################################
        

accuracy = np.sum(pred_target == y_test) / len(y_test)
print("Accuracy:", accuracy)


accuracy = np.sum(pred_target == y_test) / len(y_test)
print("Accuracy:", accuracy)
conf_matrix = confusion_matrix(y_test, pred_target)
print("Confusion Matrix:")
print(conf_matrix)

# Precision
precision = precision_score(y_test, pred_target)
print("Precision:", precision)

# Recall
recall = recall_score(y_test, pred_target)
print("Recall:", recall)

# F1-Score
f1 = f1_score(y_test, pred_target)
print("F1-Score:", f1)

# ROC-AUC Score
probabilities = clf.predict_proba(X_test)[:, 1]  # Probability of being in the positive class
roc_auc = roc_auc_score(y_test, probabilities)
print("ROC-AUC Score:", roc_auc)