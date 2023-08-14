import argparse
import os
#from glob import glob
#import random
import numpy as np
import pandas as pd
from typing import List
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
# import mlflow
from joblib import dump

# This AzureML package will allow to log our metrics etc.
from azureml.core import Dataset, Run

# Important to load in the utils as well!
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--training-data', type=str, dest='training_data', help='Reference to training dataset')
parser.add_argument('--testing-data', type=str, dest='testing_data', help='Reference to testing dataset')
parser.add_argument('--alpha', type=str, dest='alpha', help='The additive smoothing parameters to us.')
parser.add_argument('--seed', type=int, dest='seed', help='The random seed to use.')
parser.add_argument('--scoring', type=str, dest='scoring', help='The model evaluation metric to use.')
parser.add_argument('--folds', type=int, dest='folds', help='The batch size to use during training.')
parser.add_argument('--model-name', type=str, dest='model_name', help='The name of the model to use.')
args = parser.parse_args()


FOLDS = args.folds # Int
SEED = args.seed # Int
#ALPHA = args.alpha # List
ALPHA = [float(x) for x in args.alpha.split(',')] # List of float values
SCORING = args.scoring # String
MODEL_NAME = args.model_name # String

# As we're mounting the training_folder and testing_folder onto the `/mnt/data` directories, we can load in the images by using glob.
#training_paths = glob(os.path.join('./data/train', '**', 'processed_animals', '**', '*.jpg'), recursive=True)
#testing_paths = glob(os.path.join('./data/test', '**', 'processed_animals', '**', '*.jpg'), recursive=True)

# Get our context.
run = Run.get_context()
ws = run.experiment.workspace

# Get the dataset by ID
train_ds = Dataset.get_by_id(ws, id=args.training_data)
test_ds = Dataset.get_by_id(ws, id=args.testing_data)

# Load in the dataframes
# Azure Tabular datasets can be read directly into Pandas dataframes
df_train = train_ds.to_pandas_dataframe()
df_test = test_ds.to_pandas_dataframe()

# Print some examples
print("Training samples:")
print(df_train.head(5))

print("Testing samples:")
print(df_test.head(5))

# Shuffle the dataframes
#df_train = df_train.sample(frac=1, random_state=SEED)
#df_test = df_test.sample(frac=1, random_state=SEED)

# enable autologging
# mlflow.sklearn.autolog()

# Parse to Features and Targets for both Training and Testing. Refer to the Utils package for more information
X_train = df_train['product_descr']
y_train = df_train['class']

X_test = df_test['product_descr']
y_test = df_test['class']

print('Shapes:')
print(len(X_train), " - ", len(y_train))
print(len(X_test), " - ", len(y_test))


# Create an output directory where our AI model will be saved to.
# Everything inside the `outputs` directory will be logged and kept aside for later usage.
model_path = os.path.join('outputs', MODEL_NAME)
os.makedirs(model_path, exist_ok=True)

pipe = Pipeline([
       ('vect', CountVectorizer()),
       ('clf', MultinomialNB()),
])
parameters = {
#'clf__alpha': (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
#'clf__alpha': (0.1, 0.2, 0.3)
'clf__alpha': ALPHA
}
                            
grid_search = GridSearchCV(estimator = pipe, 
                           param_grid = parameters,
                           # scoring = 'accuracy',
                           # cv = 10,
                           scoring = SCORING,
                           cv = FOLDS,
                           n_jobs = -1,
                           verbose = 1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_ 
best_parameters = grid_search.best_params_  

print('Best accuracy : ', grid_search.best_score_)
print('Best parameters :', grid_search.best_params_  )

# Save the model
dump(grid_search.best_estimator_, model_path+'/gridsearch_model.joblib') 

print("Evaluating the test dataset...")
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))

class_labels = np.unique(y_test).astype(str).tolist()
cf_matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
print(cf_matrix)

# We could use this, but we are logging realtime with the callback!
# run.log_list('accuracy', history.history['accuracy'])
# run.log_list('loss', history.history['loss'])
# run.log_list('val_loss', history.history['val_loss'])
# run.log_list('val_accuracy', history.history['val_accuracy'])
# Save the metrics
run.log_list('mean_test_score', grid_search.cv_results_['mean_test_score'])
run.log_list('std_test_score', grid_search.cv_results_['std_test_score'])


## Log Confusion matrix , see https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.run?view=azure-ml-py#log-confusion-matrix-name--value--description----
cmtx = {
    "schema_type": "confusion_matrix",
    # "parameters": params,
    "data": {
        "class_labels": class_labels,
        "matrix": [[int(y) for y in x] for x in cf_matrix]
    }
}

run.log_confusion_matrix('Confusion matrix - error rate', cmtx)

# Save the confusion matrix to the outputs.
np.save('outputs/confusion_matrix.npy', cf_matrix)

print("DONE TRAINING. AI model has been saved to the outputs.")
