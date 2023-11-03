
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('churn.csv')

df.drop('RowNumber', axis=1, inplace=True)

df = df.dropna()



scaler = StandardScaler()
X = df.drop('Exited', axis=1)
X_scaled = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Exited'], test_size=0.25, random_state=42)

clf = XGBClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)

print('Accuracy:', score)