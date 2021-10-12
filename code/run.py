import numpy as np
# import csv

from implementations import (least_squares_GD, least_squares_SGD,
                             least_squares, ridge_regression,
                             logistic_regression, reg_logistic_regression)
from load_data import load_data
from data_processing import preprocessing, correctness, standardize, \
    build_model_data

# Load data
features, labels = load_data()

# Preprocessing
train_tx, train_y, test_tx, test_y = preprocessing(features, labels)
weights, loss = least_squares(train_y, train_tx)

# train_score, test_score = correctness(train_tx, train_y, test_tx, test_y, weights)

# Make prediction from test.csv
test_features, test_prediction = load_data('test')
idx = np.where(features == -999)[1]
processed_f = np.delete(features, idx, 1)
processed_l = labels
processed_f, _, _ = standardize(processed_f)
test_y, test_tx = build_model_data(processed_f, processed_l)

pred = test_tx.dot(weights)
pred[pred >= 0.5] = 1
pred[pred < 0.5] = 0

# Writing in csv file
# path_csv = "../data/sample-submission.csv"
# with open('../data/sample-submission.csv', 'w', newline='') as csvfile:
# ...


