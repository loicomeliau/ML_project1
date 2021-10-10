import numpy as np

#Functions for the calculation of the loss
def calculate_mse(e):
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return calculate_mse(e)

#Least squares regression using normal equations
def least_squares(y, tx):
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss

#Ridge regression using normal equations
def ridge_regression(y, tx, lambda_ = 0.1):
    A = tx.T.dot(tx) + 2*tx.shape[0]*lambda_*np.eye(tx.shape[1])
    x = tx.T.dot(y)
    w = np.linalg.solve(A,x)
    loss = compute_loss(y, tx, w)
    return w, loss