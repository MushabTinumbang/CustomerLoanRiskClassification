import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

print("\nCode running, please wait :)")

df = pd.read_csv('src/loandata.csv', low_memory=False)
df.term = df.term.str.extract('(\d+)').astype(int)

selected_columns = [
    'loan_amnt', 'term', 'int_rate', 'grade', 'emp_length',
    'home_ownership', 'annual_inc', 'verification_status',
    'open_acc','pub_rec','revol_bal','revol_util','total_acc', 'dti', 'loan_status'
]

df = df.loc[:, selected_columns]

le = preprocessing.LabelEncoder()

emp_length_to_int={'< 1 year':0,
                      '1 year':1,
                     '2 years':2,
                     '3 years':3,
                     '4 years':4,
                     '5 years':5,
                     '6 years':6,
                     '7 years':7,
                     '8 years':8,
                     '9 years':9,
                     '10+ years':10}
df['emp_length'] = df['emp_length'].map(emp_length_to_int)

df['loan_status'] = df['loan_status'].map({'Charged Off': 1, 'Fully Paid': 0})

df['grade']=df['grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})

df['home_ownership'] = le.fit_transform(df['home_ownership'])

df['verification_status'] = df['verification_status'].map({'Not Verified': 0, 'Verified': 1, 'Source Verified': 2 })

df = df.dropna(subset=['loan_status'])

df = df.reset_index(drop = True)
df = df.dropna()

X = df.drop('loan_status', axis=1).values
Y = df['loan_status'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Machine Learning
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1_score = metrics.f1_score(y_test, y_pred)

print("\n")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score, "\n\n")
print(classification_report(y_test, y_pred))