import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
df = pd.read_csv('/Users/sivakumarparamasivam/Sivakumar/python_workspace/ai_engineer_2025/Python Programs/ClassRoomSessions/Week_11_Day_2/sales_data.csv')

le = LabelEncoder()
df['CardType_Encoded'] = le.fit_transform(df['CardType'])

X = df[['Total Amount']]
y = df['CardType_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = LogisticRegression(class_weight='balanced', random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

Accuracy = accuracy_score(y_test, y_pred)
Precision = precision_score(y_test, y_pred, zero_division=0)
Recall = recall_score(y_test, y_pred, zero_division=0)
F1_Score = f1_score(y_test, y_pred, zero_division=0)

print(f"Accuracy : {Accuracy:.3f}")
print(f"Precision : {Precision:.3f}")
print(f"Recall : {Recall:.3f}")
print(f"F1_Score : {F1_Score:.3f}")

print("Classification Report")
print(classification_report(y_test,y_pred, target_names=le.classes_))

threshold = -model.intercept_[0] / model.coef_[0][0]
print(f"Model Threshold Card with Amount > , {threshold:.2f}")

NewValue_Predict = float (input ("Enter the total amount to classify ==>"))
input_data = pd.DataFrame({'Total Amount': [NewValue_Predict]})
predicted_index = model.predict(input_data)
predicted_card_type = le.inverse_transform(predicted_index)

print(f"Predicted Card Type :  {predicted_card_type} ")







