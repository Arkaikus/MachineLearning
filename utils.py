import json
import numpy as np
from sklearn.metrics import mean_squared_error


def read_json(name):
    return json.loads(open(name).read())


def write_json(name, a_dict):
    with open(name, 'w') as file:
        file.write(json.dumps(a_dict))


def rmse(y_train, y_pred):
    return np.sqrt(mean_squared_error(y_train, y_pred))