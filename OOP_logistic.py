import copy
import math

import numpy as np


class Logistic:
    """
    An OOP implementation to implement logistic regression to a dataset without using any ML library
    """

    @staticmethod
    def sigmoid(z):
        """
        compute the sigmoid of z
        :param z: (ndarray): A scalar, numpy array of any size.
        :return:   g (ndarray): sigmoid(z), with the same shape as z
        """
        g = 1 / (1 + np.exp(-z))
        return g

    def __init__(self, X, y, w_in, b_in):
        """
        To initialize the class Logistic
        :param X: (ndarray (m,n)): Data, m examples with n features
        :param y: (ndarray (m,)) : target values
        :param w_in: (ndarray (n,)) : model parameters
        :param b_in: (scalar)       : model parameter
        """
        self.X = X
        self.y = y
        self.w_in = w_in
        self.b_in = b_in

    def display_data(self):
        """
        To display the training data
        :return: None
        """

        print(f"Input Features : \n Size: {self.X.shape} \n Data (first 2): {self.X[:2]} \n Type: {self.X.dtype}")
        print(f"target values: \n Size: {self.y.shape} \n Data: {self.y[:2]} \n Type: {self.y.dtype}")
        print(f"w parameter {self.w_in} \n b_parameter {self.b_in}")

    def compute_cost_logistic(self, w, b):
        """
        Computes the cost for logistic regression
        :return:
            cost (scalar): cost
        """

        m = self.X.shape[0]
        cost = 0.0
        for i in range(m):
            z_i = np.dot(w, self.X[i]) + b
            f_wb = Logistic.sigmoid(z_i)
            loss = -(self.y[i] * np.log(f_wb) + (1 - self.y[i]) * np.log(1 - f_wb))
            cost += loss
        cost = cost / m
        return cost

    def compute_cost_regularized(self, w, b, lambda_=1):
        """
         Computes the cost over all examples

        Args:
          lambda_ (scalar): Controls amount of regularization
        Returns:
          total_cost (scalar):  cost
        """
        m, n = self.X.shape
        cost = 0.0
        for i in range(m):
            z_i = np.dot(w, self.X[i]) + b
            f_wb = Logistic.sigmoid(z_i)
            loss = -(self.y[i] * np.log(f_wb) + (1 - self.y[i]) * np.log(1 - f_wb))
            cost += loss
        cost = cost / m

        reg_loss = 0
        for j in range(n):
            reg_loss += w[j] ** 2
        reg_loss = (lambda_ / (2 * m)) * reg_loss

        total_cost = reg_loss + cost
        return total_cost

    def compute_gradient_logistic(self, w, b):
        """
        compute the gradient for logistic regression
        Args:
            w:
            b:
        :return:
            dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
            dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b.
        """

        m, n = self.X.shape
        dj_dw = np.zeros(n)
        dj_db = 0
        for i in range(m):
            f_wb = Logistic.sigmoid(np.dot(w, self.X[i]) + b)
            err_i = f_wb - self.y[i]
            for j in range(n):
                dj_dw_i = err_i * self.X[i, j]
                dj_dw[j] += dj_dw_i
            dj_db += err_i
        dj_db = dj_db / m
        dj_dw = dj_dw / m

        return dj_db, dj_dw

    def gradient_descent(self, alpha, num_iters):
        """
           Performs batch gradient descent

           Args:
             alpha (float)      : Learning rate
             num_iters (scalar) : number of iterations to run gradient descent

           Returns:
             w (ndarray (n,))   : Updated values of parameters
             b (scalar)         : Updated value of parameter
           """
        # An array to store cost J and w's at each iteration primarily for graphing later
        J_history = []
        w = copy.deepcopy(self.w_in)  # avoid modifying global w within function
        b = self.b_in

        for i in range(num_iters):
            # Calculate the gradient and update the parameters
            dj_db, dj_dw = Logistic.compute_gradient_logistic(self, w, b)

            # Update Parameters using w, b, alpha and gradient
            w = w - alpha * dj_dw
            b = b - alpha * dj_db

            # Save cost J at each iteration
            if i < 100000:  # prevent resource exhaustion
                J_history.append(Logistic.compute_cost_logistic(self, w, b))

            # Print cost every at intervals 10 times or as many iterations if < 10
            if i % math.ceil(num_iters / 10) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

        return w, b, J_history  # return final w,b and J history for graphing


X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  # (m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])  # (m,)
w_tmp = np.array([2., 3.])
b_tmp = 1.

logistic = Logistic(X_train, y_train, w_tmp, b_tmp)
logistic.display_data()

print(logistic.compute_cost_logistic(w_tmp, b_tmp))
print(logistic.compute_gradient_logistic(w_tmp, b_tmp))

w_out, b_out, _ = logistic.gradient_descent(0.1, 10000)
print(f" {w_out} \n {b_out}")


