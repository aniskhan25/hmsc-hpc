import numpy as np
import tensorflow as tf


def convert_to_tf(data):
    if isinstance(data, np.ndarray):
        new = tf.convert_to_tensor(data, dtype=data.dtype)
    elif isinstance(data, list):
        new = []
        for value in data:
            new.append(convert_to_tf(value))
    elif isinstance(data, dict):
        new = {}
        for key, value in data.items():
            new[key] = convert_to_tf(value)
    else:
        new = data

    return new
