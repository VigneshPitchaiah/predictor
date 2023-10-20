import numpy as np

class L1RegularizedLinearRegression:
    def __init__(self, learning_rate=0.009, num_iterations=5000, lambda_value=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_value = lambda_value
        self.theta0 = 0
        self.theta1 = 0
        self.cost_history = []
    def compute_cost_l1(self, X, y):
        m = len(y)
        y_pred = self.theta0 + self.theta1 * X
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        regularization_term = self.lambda_value * (np.abs(self.theta0) + np.abs(self.theta1))
        cost += regularization_term
        return cost
    def fit(self, X, y):
        m = len(y)
        for iteration in range(self.num_iterations):
            y_pred = self.theta0 + self.theta1 * X
            gradient_theta0 = (1 / m) * np.sum(y_pred - y)
            gradient_theta1 = (1 / m) * np.sum((y_pred - y) * X)
            reg_term_theta0 = self.lambda_value * np.sign(self.theta0)
            reg_term_theta1 = self.lambda_value * np.sign(self.theta1)
            self.theta0 -= self.learning_rate * (gradient_theta0 + reg_term_theta0)
            self.theta1 -= self.learning_rate * (gradient_theta1 + reg_term_theta1)
            cost = self.compute_cost_l1(X, y)
            self.cost_history.append(cost)
            if iteration % 50 == 0:
                print(f"Iteration {iteration}: Cost = {cost}")
    def predict(self, X):
        return self.theta0 + self.theta1 * X