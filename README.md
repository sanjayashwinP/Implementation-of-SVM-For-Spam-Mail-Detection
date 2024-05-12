# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.    
2.Read the data frame using pandas.    
3.Get the information regarding the null values present in the dataframe.    
4.Split the data into training and testing sets.    
5.Convert the text data into a numerical representation using CountVectorizer.     
6.Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.    
7.Finally, evaluate the accuracy of the model.

## Program:
/*
Program to implement the SVM For Spam Mail Detection.    
Developed by: SANJAY ASHWIN P    
RegisterNumber: 212223040181     
*/
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MARELLA HASINI
RegisterNumber:212223240083
*/
import pandas as pd
data = pd.read_csv("D:/introduction to ML/jupyter notebooks/spam.csv",encoding = 'windows-1252')
from sklearn.model_selection import train_test_split
data
data.shape
x = data['v2'].values
y = data['v1'].values
x.shape
y.shape
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.35,random_state = 48)
x_train
x_train.shape
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc = accuracy_score(y_test,y_pred)
acc
con = confusion_matrix(y_test,y_pred)
print(con)
cl = classification_report(y_test,y_pred)
print(cl)
```

## Output:
### Result Output
![image](https://github.com/sanjayashwinP/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473265/318fbdd7-38f6-431a-834e-118824154e33)

![image](https://github.com/sanjayashwinP/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473265/067683a5-6e87-42f4-ab60-fe61dfc90c4f)

![image](https://github.com/sanjayashwinP/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473265/a8210773-90cb-45e3-834c-edce3370a9cb)

![image](https://github.com/sanjayashwinP/Implementation-of-SVM-For-Spam-Mail-Detection/assets/147473265/8afd486f-4ff3-44bf-8d3c-32c46b7e38e6)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
