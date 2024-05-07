# EX:09 Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.


## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Syed Mokthiyar S.M
RegisterNumber: 212222230156
```
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('/content/spam.csv', encoding='ISO-8859-1')
df.head()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['v2'])
y = df['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = svm.SVC (kernel='linear') 
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy: ", accuracy_score (y_test, predictions)) 
print("Classification Report: ")
print(classification_report (y_test, predictions))
```

## Output:
## DATASET :
![Screenshot 2024-04-30 141904](https://github.com/chandrumathiyazhagan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393023/def58994-8736-47f5-85cb-93a854173ff6)

## Kernel Model:
![Screenshot 2024-04-30 141911](https://github.com/chandrumathiyazhagan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393023/16a5bdc0-c177-4315-8245-eb75bddb7ede)

## Accuracy and Classification Report :  
![Screenshot 2024-04-30 141918](https://github.com/chandrumathiyazhagan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393023/d95c6069-fd58-4c92-939f-a21412ea5941)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
