import os
import numpy as np
import json
from joblib import load

def init():
    global model

    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_path = os.path.join(os.environ.get('AZUREML_MODEL_DIR'), 'mro-model_no', 'gridsearch_model.joblib')
    print("Loading model from path:", model_path)

    model = load(model_path)


def run(input_data):
    product = json.loads(input_data)["product"]
    data = np.array([product])
    pred_class = model.predict(data)[0]

    return {"class": pred_class}