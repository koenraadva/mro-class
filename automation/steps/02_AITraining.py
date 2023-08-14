#from ctypes import resize
#from glob import glob
#import json
import os
#from datetime import datetime
#import math
#import random
#import shutil
from typing import List, Tuple

from utils import connectWithAzure

from azureml.core import ScriptRunConfig, Experiment, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.environment import Environment
#from azureml.core.conda_dependencies import CondaDependencies

from dotenv import load_dotenv

# When you work locally, you can use a .env file to store all your environment variables.
# This line read those in.
load_dotenv()

MRO_LANG = os.environ.get('MRO_LANG').split(',')
SEED = int(os.environ.get('RANDOM_SEED'))
TRAIN_ON_LOCAL = os.environ.get('TRAIN_ON_LOCAL').lower() in ('true', '1', 't')

FOLDS = int(os.environ.get('FOLDS'))
ALPHA = os.environ.get('ALPHA')
SCORING = os.environ.get('SCORING')
MODEL_NAME = os.environ.get('MODEL_NAME')

COMPUTE_NAME = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpu-cluster")
COMPUTE_MIN_NODES = int(os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0))
COMPUTE_MAX_NODES = int(os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4))

# This example uses CPU VM. For using GPU VM, set SKU to STANDARD_NC6
VM_SIZE = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")

def prepareComputeCluster(ws):
    if COMPUTE_NAME in ws.compute_targets:
        compute_target = ws.compute_targets[COMPUTE_NAME]
        if compute_target and type(compute_target) is AmlCompute:
            print("found compute target: " + COMPUTE_NAME)
    else:
        print("creating new compute target...")
        provisioning_config = AmlCompute.provisioning_configuration(vm_size = VM_SIZE,
                                                                    min_nodes = COMPUTE_MIN_NODES, 
                                                                    max_nodes = COMPUTE_MAX_NODES)

        # create the cluster
        compute_target = ComputeTarget.create(ws, COMPUTE_NAME, provisioning_config)
        
        # can poll for a minimum number of nodes and for a specific timeout. 
        # if no min node count is provided it will use the scale settings for the cluster
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
        
        # For a more detailed view of current AmlCompute status, use get_status()
        print(compute_target.get_status().serialize())

    return compute_target

def prepareEnvironment(ws):

    # Create an Environment name for later use
    environment_name = os.environ.get('TRAINING_ENV_NAME')
    conda_dependencies_path = os.environ.get('CONDA_DEPENDENCIES_PATH')

    if TRAIN_ON_LOCAL:
        # Create an environment for training on local machine
        env = Environment(environment_name) 
        env.python.user_managed_dependencies = True
    else:
        # We can directly create an environment from a saved file
        env = Environment.from_conda_specification(environment_name, file_path=conda_dependencies_path)
        env.python.user_managed_dependencies = False

    # Register environment to re-use later
    env.register(workspace = ws)

    return env

def prepareTraining(ws, env, compute_target, mro_lang) -> Tuple[Experiment, ScriptRunConfig]:
    experiment_name = os.environ.get('EXPERIMENT_NAME')
    script_folder = os.environ.get('SCRIPT_FOLDER')

    train_set_name = os.environ.get('TRAIN_SET_NAME')+"_"+mro_lang[:2]
    test_set_name = os.environ.get('TEST_SET_NAME')+"_"+mro_lang[:2]

    datasets = Dataset.get_all(workspace=ws) # Get all the datasets
    exp = Experiment(workspace=ws, name=experiment_name) # Create a new experiment

    args = [
    # Provide a name for this dataset which will be used to retrieve the materialized dataset in the run
    '--training-data', datasets[train_set_name].as_named_input('train_ds'), # Currently, this will always take the last version. You can search a way to specify a version if you want to
    '--testing-data', datasets[test_set_name].as_named_input('test_ds'), # Currently, this will always take the last version. You can search a way to specify a version if you want to
    '--alpha', ALPHA,
    '--seed', SEED,
    '--folds', FOLDS,
    '--scoring', SCORING,
    '--model-name', MODEL_NAME+"_"+mro_lang[:2]]

    script_run_config = ScriptRunConfig(source_directory=script_folder,
                    script='train.py',
                    arguments=args,
                    compute_target=compute_target,
                    environment=env)


    return exp, script_run_config

def downloadAndRegisterModel(ws, run, mro_lang):
    model_path = 'outputs/' + MODEL_NAME+"_"+mro_lang[:2]

    datasets = Dataset.get_all(workspace=ws) # Get all the datasets
    test_set_name = os.environ.get('TEST_SET_NAME')+"_"+mro_lang[:2]

    run.download_files(prefix=model_path)
    run.register_model(MODEL_NAME+"_"+mro_lang[:2],
                model_path=model_path,
                tags={'language': mro_lang, 'AI-Model': 'NB', 'GIT_SHA': os.environ.get('GIT_SHA')},
                description="MRO classification using Naive Bayes",
                sample_input_dataset=datasets[test_set_name])

def main():
    ws = connectWithAzure()

    if TRAIN_ON_LOCAL:
        compute_target = None
    else:
        compute_target = prepareComputeCluster(ws)
   
    environment = prepareEnvironment(ws)

    for mro_lang in MRO_LANG: 
        exp, config = prepareTraining(ws, environment, compute_target, mro_lang)

        run = exp.submit(config=config)
        print(f"Run started for {mro_lang} language!")
        # Run output can be turned off, you can follow that on the Azure logs if you want to.
        run.wait_for_completion(show_output=False)
        print(f"Run {run.id} has finished.")

        downloadAndRegisterModel(ws, run, mro_lang)
    
    print("All runs have finished.")

if __name__ == '__main__':
    main()