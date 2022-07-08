""" Name:Rajeev Kumar
    Roll:B20124
    mo:9341062431
"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
test_df=pd.read_csv("SteelPlateFaults-test.csv")
train_df=pd.read_csv("SteelPlateFaults-train.csv")
Y_true=test_df["Class"]
test_df.drop(columns=['Unnamed: 0',"X_Minimum", "Y_Minimum", "TypeOfSteel_A300", "TypeOfSteel_A300","Class" ],inplace=True)
train_df0=train_df[train_df.Class == 0]
train_df1=train_df[train_df.Class == 1]
train_df0.drop(columns=['Unnamed: 0',"X_Minimum", "Y_Minimum", "TypeOfSteel_A300", "TypeOfSteel_A300","Class" ],inplace=True)
train_df1.drop(columns=['Unnamed: 0',"X_Minimum", "Y_Minimum", "TypeOfSteel_A300", "TypeOfSteel_A300","Class" ],inplace=True)
train_mean0=train_df0.mean()
train_mean1=train_df1.mean()
print(train_mean0,train_mean1)
def calculateCovariance(X):
    meanX = np.mean(X, axis = 0)
    lenX = X.shape[0]
    X = X - meanX
    covariance = X.T.dot(X)/lenX
    return covariance
train_cov0=calculateCovariance(train_df0)
train_cov1=calculateCovariance(train_df1)
print(train_cov0.to_string())
print(train_cov1.to_string())
t1=train_df1.subtract(train_mean1, axis = 1)
Y_pred=[]
#print(test_df.iloc[0],train_mean1)
#print(test_df.iloc[0].subtract(train_mean1).T)
for i in range(336):
    t0=(np.exp(-0.5*np.dot(np.dot((test_df.iloc[i].subtract(train_mean0)).T, np.linalg.inv(train_cov0)),
                                  ((test_df.iloc[i].subtract(train_mean0))))))/((2*np.pi)*5 * (np.linalg.det(train_cov0))*0.5)
    t1=(np.exp(-0.5*np.dot(np.dot((test_df.iloc[i].subtract(train_mean1)).T, np.linalg.inv(train_cov1)),
                                  ((test_df.iloc[i].subtract(train_mean1))))))/((2*np.pi)*5 * (np.linalg.det(train_cov1))*0.5)
    if t1>t0:
        Y_pred.append(1)
    else:
        Y_pred.append(0)
cm = confusion_matrix(Y_true, Y_pred)
print(cm)
print("Accuracy:",metrics.accuracy_score(Y_true, Y_pred))