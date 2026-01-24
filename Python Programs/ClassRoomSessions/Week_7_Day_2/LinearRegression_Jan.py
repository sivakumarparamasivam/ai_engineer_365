import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])
model = LinearRegression()
model.fit(X,y)
print("Slope (coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
X_new =np.array([6]).reshape(-1,1)
try:
	user_input = input("Enter a value to predict: ").strip()
	val = float(user_input)
	prediction = model.predict([[val]])[0]
	print(f"Prediction for {val} -> {prediction}")
except Exception as e:
	print("Invalid input or prediction error:", e)

