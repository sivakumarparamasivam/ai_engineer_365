import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



data_dict = {
    'x1': [1, 2, 3],
    'x2': [2, 1, 4],
    'Output': [6, 8, 14]
}
data = pd.DataFrame(data_dict)


X = data[['x1', 'x2']]  
y = data['Output']     


X_train = X
X_test = X
y_train = y
y_test = y


model = LinearRegression()
model.fit(X_train, y_train)


print("--- Model Coefficients ---")
print(f"Intercept (β0): {model.intercept_:.1f}")
print(f"Coefficients (β1, β2): {model.coef_}")
print(f"Model Equation: y = {model.intercept_:.1f} + {model.coef_[0]:.1f}*x1 + {model.coef_[1]:.1f}*x2\n")


y_pred = model.predict(X_test)


print("--- Model Performance Evaluation ---")
print(f"Predicted values: {y_pred}")
print(f"Actual values:    {y_test.values}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}") 
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(X_test['x1'], X_test['x2'], y_test, color='blue', s=100, label='Actual Data Points')


x1_surf = np.linspace(X['x1'].min(), X['x1'].max(), 10)
x2_surf = np.linspace(X['x2'].min(), X['x2'].max(), 10)
x1_surf, x2_surf = np.meshgrid(x1_surf, x2_surf)


y_plane = model.predict(pd.DataFrame({'x1': x1_surf.ravel(), 'x2': x2_surf.ravel()}))
y_plane = y_plane.reshape(x1_surf.shape)


ax.plot_surface(x1_surf, x2_surf, y_plane, color='red', alpha=0.3, label='Predicted Plane')

ax.set_xlabel('x1 (Total Years of Experience - equivalent)')
ax.set_ylabel('x2 (Rating)')
ax.set_zlabel('Salary')
ax.set_title('Salary Prediction (3D Regression Plane)')
ax.legend()

plt.show()

print('\n Predict the value')

try:
    x1 = int(input('Enter x1 value: '))
    x2 = int(input('Enter X2 value: '))

    new_test_data = pd.DataFrame({'x1':[x1], 'x2':[x2]
                                  })
    
    predicted_value = model.predict(new_test_data)

    print('Predicted Value:', predicted_value)

except Exception as e:
    print('Error')    