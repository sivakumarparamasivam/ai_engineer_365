import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.array([1,2,3,4]).reshape(-1,1)
y = np.array([1,4,9,16])

poly = PolynomialFeatures(degree=3)
Xpoly = poly.fit_transform(X)

model = LinearRegression()
model.fit(Xpoly, y)

print(f" Coef {model.coef_}")
print(f"Intercept {model.intercept_:.1f}")

y_predict = model.predict(Xpoly)
print("Predictions on training data:", y_predict)

user_input = float(input("Enter a value: "))

user_input_poly = poly.transform([[user_input]])
print('Polinomial values ',user_input_poly)
prediction = model.predict(user_input_poly)

print(f"Predicted value for {user_input} =>  {prediction[0]}")

plt.scatter(X,y, color='blue', label ='Original Data')
plt.plot(X, y_predict, color='red', label='Polynomial Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression Example')
plt.legend()
plt.show()






