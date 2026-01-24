import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

rawData = pd.read_csv('/Users/sivakumarparamasivam/Sivakumar/python_workspace/ai_engineer_2025/Python Programs/Assignments_Python/2026/Jan_17_Assignments/sales_data.csv')

plt.scatter(rawData['Advertising_Spend'], rawData['Total Amount'], color='#1f77b4', label='Actual Sales Data', s=50, alpha=0.6)
plt.xlabel('Advertising Spend')
plt.ylabel('Total Amount')
plt.title('Sales vs Advertising Spend')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

sns.histplot(rawData['Advertising_Spend'], color='#ff7f0e', kde=True, stat='density', label='Distribution')
plt.legend()
plt.show()

#Polynomial Regression does not strictly require feature scaling, as it is based on Linear Regression. However, scaling is often applied to improve numerical stability when polynomial features result in large values. In this assignment, scaling was optional due to the use of a single feature and low polynomial degree.
adv_Q1 = rawData['Advertising_Spend'].quantile(0.25)
adv_Q3 = rawData['Advertising_Spend'].quantile(0.75)

adv_IQR = adv_Q3 - adv_Q1
lower_limit = adv_Q1 - 1.5 * adv_IQR
upper_limit = adv_Q3 + 1.5 * adv_IQR

adv_outliers_removed = rawData[(rawData['Advertising_Spend'] > lower_limit) & (rawData['Advertising_Spend'] < upper_limit)]
print(adv_outliers_removed)
sns.histplot(adv_outliers_removed, x=adv_outliers_removed['Advertising_Spend'], bins=10, kde=True, color='#2ca02c', label='After Outlier Removal')
plt.title('Advertising Spend (Outliers Removed)')
plt.xlabel('Advertising Spend')
plt.ylabel('Frequency')
plt.legend()
plt.show()

X = adv_outliers_removed['Advertising_Spend'].to_numpy().reshape(-1,1)
Y = adv_outliers_removed['Total Amount'].to_numpy()

xTrain, xTest, yTrain, yTest = train_test_split(X,Y,test_size=0.2,random_state=10)
poly = PolynomialFeatures(degree=3)

xTrainPoly = poly.fit_transform(xTrain)
xTestPoly = poly.fit_transform(xTest)

model = LinearRegression()
model.fit(xTrainPoly, yTrain)

yPredit = model.predict(xTestPoly)
sorted_idx = xTest.flatten().argsort()

poly_mse = mean_squared_error(yTest,yPredit)
poly_rmse = root_mean_squared_error(yTest,yPredit)
poly_r2 = r2_score(yTest,yPredit)


model2 = LinearRegression()
model2.fit(xTrain, yTrain)

yPreditLinear = model2.predict(xTest)

# Create subplots for Polynomial and Linear Regression
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sorted_idx = xTest.flatten().argsort()
ax1.scatter(xTest, yTest, color='#d62728', label='Actual', alpha=0.6, s=50, marker='o')
ax1.plot(xTest.flatten()[sorted_idx], yPredit[sorted_idx], color='#1f77b4', label='Predicted', linewidth=2.5, marker='o')
ax1.set_title('Polynomial Regression (Degree 3)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Advertising Spend', fontsize=11)
ax1.set_ylabel('Total Amount (Sales)', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)

sorted_idx = xTest.flatten().argsort()
ax2.scatter(xTest, yTest, color='#d62728', label='Actual', alpha=0.6, s=50, marker='o')
ax2.plot(xTest.flatten()[sorted_idx], yPreditLinear[sorted_idx], color='#2ca02c', label='Predicted', linewidth=2.5, marker='o')
ax2.set_title('Linear Regression', fontsize=12, fontweight='bold')
ax2.set_xlabel('Advertising Spend', fontsize=11)
ax2.set_ylabel('Total Amount (Sales)', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

linear_mse = mean_squared_error(yTest, yPreditLinear)
linear_rmse = root_mean_squared_error(yTest, yPreditLinear)
linear_r2 = r2_score(yTest, yPreditLinear)
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'R² Score'],
    'Polynomial Regression': [poly_mse, poly_rmse, poly_r2],
    'Linear Regression': [linear_mse, linear_rmse, linear_r2]
})

print("\n" + "="*60)
print("EVALUATION METRICS COMPARISON")
print("="*60)
print(metrics_df.to_string(index=False))
print("="*60)

print("\n" + "="*60)
print("PREDICTION WITH USER INPUT")
print("="*60)
user_input = float(input("Enter Advertising Spend amount: "))

user_input_reshaped = np.array([[user_input]])

poly_features = PolynomialFeatures(degree=3)
user_input_poly = poly_features.fit_transform(user_input_reshaped)
poly_prediction = model.predict(user_input_poly)[0]

linear_prediction = model2.predict(user_input_reshaped)[0]

print(f"\nFor Advertising Spend: ${user_input:.2f}")
print("-" * 60)
print(f"Polynomial Regression Prediction: ${poly_prediction:.2f}")
print(f"Linear Regression Prediction: ${linear_prediction:.2f}")
print("="*60)