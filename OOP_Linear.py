import numpy as np

import copy
import math


def predict(x, w, b):
    """
    single predict using linear regression
    :param x:  (ndarray): Shape (n,) example with multiple features
    :param w, b: model parameters
    :return:
        p (scalar): prediction
    """
    p = np.dot(x, w) + b
    return p


class Linear:
    """
    Methods to implement Linear regression algorithms without using any ML library
    """

    def __init__(self, X, y, w_in, b_in):
        """
        initialize the Linear class
        :param X: (ndarray) : Data m, examples
        :param y: (ndarray) : Output set(target values)
        :param w_in: (ndarray) : Model parameters
        :param b_in: (scalar) : Model parameter
        """

        self.X = X
        self.y = y
        self.w_in = w_in
        self.b_in = b_in

    def display_data(self):
        """
        Display the training data
        :return: None
        """
        print(f"Input Features : \n Size: {self.X.shape} \n Data (first 2): {self.X[:2]} \n Type: {self.X.dtype}")
        print(f"target values: \n Size: {self.y.shape} \n Data: {self.y[:2]} \n Type: {self.y.dtype}")
        print(f"w parameter {self.w_in} \n b_parameter {self.b_in}")

    def compute_cost(self, w, b):
        """
        computes the  cost function foe linear regression
        :param w : (ndarray) : Model parameters
        :param b: (scalar) : Model parameters
        :return:
            total_cost (float) : the total cost using w,b as model parameters
        """
        m = self.X.shape[0]
        cost_sum = 0.0
        for i in range(m):
            f_wb_i = np.dot(self.X[i], w) + b
            err = (f_wb_i - self.y[i]) ** 2
            cost_sum += err
        total_cost = cost_sum * (1 / (2 * m))

        return total_cost

    def compute_gradient(self, w, b):
        """
        Computes the gradient for linear regression

        :param w: (ndarray) : model parameters
        :param b: (scalar) : Model parameters
        :return:
            dj_dw: (scalar) : gradient wrt to w
            dj_db; (scaler) : gradient wrt to b
        """

        m, n = self.X.shape  # (number of examples, number of features)
        dj_db = 0
        dj_dw = np.zeros(n)
        for i in range(m):
            err = (np.dot(self.X[i], w) + b) - self.y[i]
            for j in range(n):
                dj_dw_i = err * self.X[i, j]
                dj_dw[j] += dj_dw_i
            dj_db += err
        dj_dw = dj_dw / m
        dj_db = dj_db / m

        return dj_dw, dj_db

    def gradient_descent(self, learn, num_iter):
        """
        Performs gradient descent to fit w,b. Updates w,b by taking
        num_iters gradient steps with learning rate alpha

        :param learn: (scalar) : Learning rate
        :param num_iter: (scaler) : number of iterations to run gradient descent
        :return:
          w (scalar): Updated value of parameter after running gradient descent
          b (scalar): Updated value of parameter after running gradient descent
          J_history (List): History of cost values
        """

        J_history = []
        b = self.b_in
        w = copy.deepcopy(self.w_in)  # avoid modifying global w within function

        for i in range(num_iter):

            # Calculate the gradient and update the parameters using gradient_function
            dj_dw, dj_db = Linear.compute_gradient(self, w, b)

            # Update Parameters using equation (3) above
            w = w - learn * dj_dw
            b = b - learn * dj_db

            # Save cost j after each iteration
            if i < 100000:  # prevent resource exhaustion
                J_history.append(Linear.compute_cost(self, w, b))

            # print cost after intervals of 10 times
            if i % math.ceil(num_iter / 10) == 0:
                print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

        return w, b, J_history  # for plotting


# load training data
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])  # note 4 input features
y_train = np.array([460, 232, 178])
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7

linear = Linear(X_train, y_train, initial_w, initial_b)
linear.display_data()

# get a row from our training data
x_vec = X_train[0, :]
f_wb = predict(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

cost = linear.compute_cost(w_init, b_init)
print(cost)

# Compute and display gradient
tmp_dj_db, tmp_dj_dw = linear.compute_gradient(w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')


# run gradient descent
w_final, b_final, _ = linear.gradient_descent(alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")


