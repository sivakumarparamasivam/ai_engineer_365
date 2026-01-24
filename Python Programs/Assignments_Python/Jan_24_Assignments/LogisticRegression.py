import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, classification_report)
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
# CSV File Load & Basic validations
base_dir = Path(__file__).parent/"Social_Network_Ads.csv"

df= pd.read_csv(base_dir)
print(df.head(10))
print("Data Shape", df.shape)
print("Data Describe", df.describe())
print("Checking missing value", df.isnull().sum().sum())

# Data Training

le = LabelEncoder()
df['Purchased_Feature_Encoded'] = le.fit_transform(df['Purchased'])
X = df[['Age','EstimatedSalary']]
y = df['Purchased_Feature_Encoded']

model = LogisticRegression(class_weight='balanced',random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix")
cm=confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=['Actual No', 'Actual Yes'],
    columns=['Pred No', 'Pred Yes']
)

print(cm_df)
labels = ['Not Purchased', 'Purchased']
plt.figure(figsize=(5,4))
plt.imshow(cm)   # no color specified
plt.colorbar()

# classes = le.classes_  # or ['No','Yes']

# plt.xticks(range(len(classes)), classes)
# plt.yticks(range(len(classes)), classes)

# Axis ticks with class names
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

# write numbers inside boxes
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

Accuracy = accuracy_score(y_test, y_pred)
Precision = precision_score(y_test, y_pred, zero_division=0)
Recall = recall_score(y_test, y_pred, zero_division=0)
F1_Score = f1_score(y_test, y_pred, zero_division=0)

print(f"Accuracy : {Accuracy:.3f}")
print(f"Precision : {Precision:.3f}")
print(f"Recall : {Recall:.3f}")
print(f"F1_Score : {F1_Score:.3f}")

print("Classification Report")
print(classification_report(y_test,y_pred, target_names=['No', 'Yes']))
print("*"*100)
print("Prediction Probability:\n", model.predict_proba(X_test))

# ---- Take inputs ----
age = float(input("Enter Age: "))
salary = float(input("Enter Estimated Salary: "))

# ---- Prepare input for model (2D array required) ----
sample = np.array([[age, salary]])

# ---- Probability prediction ----
prob = model.predict_proba(sample)[0][1]   # class 1 probability

# ---- Final class prediction ----
pred = model.predict(sample)[0]

label = "Purchased" if pred == 1 else "Not Purchased"

# ---- Display results ----
print("\n===== Prediction Result =====")
print(f"Purchase Probability : {prob:.2f}")
print(f"Final Prediction    : {label}")


