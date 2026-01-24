import numpy as np
from sklearn.linear_model import Ridge

X=np.array([[1,1],[1,2],[1,3]])
y=np.array([1,2,2])

alpha = 1.0

model = Ridge(alpha=alpha, fit_intercept=False)
model.fit(X, y)

print(f"Intercept ", model.intercept_)
print(f"Co-efficient ", model.coef_)

try:
    user_input = float(input("\nEnter the value to predict: "))
    input_2d = np.array([[user_input, user_input]])
    predict_value = model.predict(input_2d)[0]
    print(f"Predicted Value: {predict_value:.4f}")
except ValueError:
    print("Enter a valid numerical value to predict")    