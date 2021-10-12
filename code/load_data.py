import numpy as np


def load_data():
    path_dataset = "../../data/train.csv"
    feature_cols = np.arange(32)[2:31]
    data = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=feature_cols)
    features = data
    labels = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1], dtype=bytes,
        converters={1: lambda x: 0 if b"s" in x else 1})
    return features, labels

