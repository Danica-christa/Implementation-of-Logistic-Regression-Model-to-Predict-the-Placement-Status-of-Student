# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and.duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. Apply new unknown values

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: Danica Christa
RegisterNumber: 212223240022

import pandas as pd
df=pd.read_csv("Placement_Data.csv")
print(df.head())

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
print(df1.head())

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
print(df1)

x=df1.iloc[:,:-1]
print(x)

y=df1["status"]
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

## PLACEMENT DATA:

![alt text](<Screenshot 2024-04-20 185425.png>)
## SALARY DATA:

![alt text](<Screenshot 2024-04-20 185528.png>)
## NULL DATA:

![alt text](<Screenshot 2024-04-20 185615.png>)
## DATA DUPLICATE:

![alt text](<Screenshot 2024-04-20 190115.png>)
## PRINT DATA:

![alt text](<Screenshot 2024-04-20 190507.png>)
## DATA_STATUS:

![alt text](<Screenshot 2024-04-20 190405.png>)
## DATA_STATUS:

![alt text](<Screenshot 2024-04-20 190618.png>)
## Y_PREDICTION ARRAY:

![alt text](<Screenshot 2024-04-20 190645.png>)
## ACCURACY VALUE:

![alt text](<Screenshot 2024-04-20 190744.png>)
## CONFUSION ARRAY:

![alt text](<Screenshot 2024-04-20 191238.png>)
## CLASSIFICATION REPORT:

![alt text](<Screenshot 2024-04-20 191550.png>)
## PREDICTION OF LR:

![alt text](<Screenshot 2024-04-20 191613.png>)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
