""" Name:Rajeev Kumar
    Roll:B20124
    mo:9341062431
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
content=pd.read_csv("SteelPlateFaults-2class.csv")
X = content.iloc[:, :-1].values
y = content.iloc[:, 27].values
[X_train, X_test, X_label_train, X_label_test] =train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True)
#Renaming axis in new csv file
colum=[]
for col in content:
    colum.append(col)
train=pd.DataFrame(X_train)
test=pd.DataFrame(X_test)
for col in train :
    max=train[col].max()
    min=train[col].min()
    for k in range(len(train[col])):
        train[col].iloc[k] = (train[col].iloc[k] - min) / (max - min)
    for k in range(len(test[col])):
        p= (test[col].iloc[k] - min) / (max - min)
        if(p>1 ):
            test[col].iloc[k]=1
        elif(p<0):
            test[col].iloc[k] =0
        else:
            test[col].iloc[k]=p
X_train=train
X_test=test
#forming new csv file
train['Class']=X_label_train
train.columns=colum
test['Class']=X_label_test
test.columns=colum
train.to_csv("SteelPlateFaults-train-Normalised.csv")
test.to_csv("SteelPlateFaults-test-Normalised.csv")
for item in[1,3,5]:
    print("For k=",item)
    #Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=item)
    #Train the model using the training sets
    knn.fit(X_train, X_label_train)
    #Predict the response for test dataset
    y_pred = knn.predict(X_test)
    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy
    print("Accuracy:",metrics.accuracy_score(X_label_test, y_pred))
    cm = confusion_matrix(X_label_test, y_pred)
    print(cm)