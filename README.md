# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start 
2.Import Necessary Libraries and Load Data
3.Split Dataset into Training and Testing Sets
4.Train the Model Using Stochastic Gradient Descent (SGD)
5.Make Predictions and Evaluate Accuracy
6.Generate Confusion Matrix
7.End

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: DIVADHARSHINI.A
RegisterNumber: 212222240027

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df.head())
x=df.drop('target',axis=1)
x
y=df['target']
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred=sgd_clf.predict(x_test)
print("Prediction")
y_pred
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy:.3f}")
confusion=confusion_matrix(y_test,y_pred)
print("confusion matrix")
confusion

*/
```

## Output:

![image](https://github.com/user-attachments/assets/0c8730e1-edd6-4fcd-9941-8c335c2063a6)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
