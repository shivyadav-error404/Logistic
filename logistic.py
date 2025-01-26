import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

X = pd.read_csv(r'C:\Users\KIIT\Desktop\ML\logisticX.csv', header=None).values
y = pd.read_csv(r'C:\Users\KIIT\Desktop\ML\logisticY.csv', header=None).values.ravel()

X_with_intercept = sm.add_constant(X)
logit_model = sm.Logit(y, X_with_intercept)
result = logit_model.fit()

coefficients = result.params
intercept = coefficients[0]
cost_function_value = result.llf

print("Coefficients:", coefficients[1:])
print("Intercept:", intercept)
print("Cost Function Value (Log-Likelihood):", cost_function_value)

def gradient_descent_cost(X, y, learning_rate=0.1, max_iter=50):
    m, n = X.shape
    theta = np.zeros(n + 1)
    X_with_bias = np.c_[np.ones((m, 1)), X]
    costs = []
    for _ in range(max_iter):
        z = X_with_bias @ theta
        h = 1 / (1 + np.exp(-z))
        gradient = (1 / m) * X_with_bias.T @ (h - y)
        theta -= learning_rate * gradient
        cost = -(1 / m) * np.sum(y * np.log(h) - (1 - y) * np.log(1 - h))
        costs.append(cost)
    return costs

costs = gradient_descent_cost(X, y, learning_rate=0.1, max_iter=50)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 51), costs, marker='o', label="Cost Function")
plt.title("Cost Function vs. Iterations", fontsize=14)
plt.xlabel("Iterations", fontsize=12)
plt.ylabel("Cost Function Value", fontsize=12)
plt.grid()
plt.legend()
plt.show()


def plot_decision_boundary(X, y, coefficients):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0', alpha=0.7)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1', alpha=0.7)
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = -(coefficients[0] + coefficients[1] * x1) / coefficients[2]
    plt.plot(x1, x2, color='green', label='Decision Boundary', linewidth=2)
    plt.title("Logistic Regression Decision Boundary", fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

plot_decision_boundary(X, y, coefficients)

X_new = np.c_[X, X[:, 0] ** 2, X[:, 1] ** 2]
X_new_with_intercept = sm.add_constant(X_new)

logit_model_new = sm.Logit(y, X_new_with_intercept)
result_new = logit_model_new.fit()

coefficients_new = result_new.params
plot_decision_boundary(X_new[:, :2], y, coefficients_new)

y_pred_prob = result.predict(X_with_intercept)
y_pred = (y_pred_prob >= 0.5).astype(int)
conf_matrix = confusion_matrix(y, y_pred)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
