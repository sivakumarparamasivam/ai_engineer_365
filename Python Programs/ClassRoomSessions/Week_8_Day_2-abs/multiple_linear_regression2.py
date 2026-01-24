import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the csv file
try:
    data =pd.read_csv("/Users/sivakumarparamasivam/Sivakumar/python_workspace/ai_engineer_2025/Python Programs/ClassRoomSessions/Week_8_Day_2-abs/salary.csv")
except Exception as e:
    print('error',e)
X = data[['YearsExperience', 'Rating']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print('Model Co-Efficients')
print(f"Intercept (Beta 0): {model.intercept_:.2f}")
print(f"Co-efficiencts (Beta_Exp, Beta_Rating): {model.coef_}")

if len(model.coef_)==2:
    print(f"Model Equation: y= {model.intercept_:.2f} + {model.coef_[0]:.2f} * YearsExperience + {model.coef_[1]:.2f} * Salary")
else:
    print('Model equation not displayed due to co-efficient mismatch. \n')  


y_pred = model.predict(X_test)

print(y_pred)

# Evaluation
print("Model Evaluation")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred): .2f}")
print(f"R Square : {r2_score(y_test, y_pred): .2f}")

# Plotting

fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test['YearsExperience'], X_test['Rating'], y_test, color='blue', s=100, label = 'Actual Test Data')

x1_surf = np.linspace(X['YearsExperience'].min(), X['YearsExperience'].max(), 10)
x2_surf = np.linspace(X['Rating'].min(), X['Rating'].max(), 10)
x1_surf, x2_surf = np.meshgrid(x1_surf, x2_surf)

print(x1_surf, x2_surf)

y_plane = model.predict(pd.DataFrame({'YearsExperience': x1_surf.ravel(), 'Rating':x2_surf.ravel()}))
y_plane = y_plane.reshape(x1_surf.shape)

ax.plot_surface(x1_surf, x2_surf, y_plane, color = 'red', alpha = 0.3, label = 'Predicted Plane')
ax.set_ylabel('Rating')
ax.set_zlabel('Salary')
ax.set_title('Salary Prediction (3D Regression Plane)')
ax.legend()
plt.show()

yoe = int(input("Enter the Years of Experience: "))
rating = int(input("Enter the Ratings: "))

new_test_data = pd.DataFrame({'Years Of Experience':[yoe], 'Rating':[rating]
                                  })
    
predicted_value = model.predict(new_test_data)

print('Predicted Value:', predicted_value)


