# Logistic Regression and Its Metrics

## 1. Why is Logistic Regression suitable for this problem?
Logistic Regression is suitable for binary classification tasks, such as predicting whether a user will click on an ad based on features like age, salary, and gender. It estimates the probability of a binary outcome using a logistic function, making it effective for problems where the relationship between the independent variables and the dependent variable is approximately linear in the log-odds space. Additionally, it provides interpretable results, allowing businesses to understand the impact of different features on the prediction.

## 2. What does Precision indicate in a business context?
Precision measures the proportion of true positive predictions among all positive predictions made by the model. In a business context, high precision indicates that when the model predicts a positive outcome (e.g., a user will click on an ad), it is likely to be correct. This is particularly important in scenarios where false positives can lead to wasted resources, such as in targeted advertising or fraud detection.

## 3. What does Recall indicate in a business context?
Recall measures the proportion of true positive predictions among all actual positive cases. In a business context, high recall means that the model successfully identifies most of the actual positive cases (e.g., users who are likely to click on an ad). This is crucial in situations where missing a positive case can result in lost opportunities, such as in customer retention strategies or identifying potential leads.

## 4. If Precision is high but Recall is low, what does it mean?
A high precision but low recall indicates that while the model is very accurate when it predicts a positive outcome, it fails to identify many actual positive cases. This situation can occur when the model is overly conservative, leading to many false negatives. In a business context, this could mean that the company is effectively targeting a small group of likely customers but is missing out on a larger group of potential customers who would also be interested in the product or service.
