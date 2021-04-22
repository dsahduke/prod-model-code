# Setup
import argparse
import json
import logging
import numpy as np
import pandas as pd
import re
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

# Set up the logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)

# Set up command line arguments
parser = argparse.ArgumentParser(description="Train either a logistic regression or random forest on the diabetes data")
parser.add_argument('-d', '--debug', help="Run the logger in debug mode to get more verbose messages", action="store_true")
parser.add_argument('model', help="One of rf or lr, specifying which model to run", type=str)
args = parser.parse_args()

logging.info(f"Now running and training with {args.model}")

# Creating constants
ENCOUNTER_ID = 'encounter_id'
RACE = 'race'
GENDER = 'gender'
NUM_LABS = 'num_lab_procedures'
READMISSION = 'readmission'
DATA_PATH = "./data/diabetes_df.csv"


first_number_regex = re.compile(r'^\[(\d*)') # Matches a bracket followed by numbers (0 or more, greedy)

def extract_minimum_age(age):
    matching_number = first_number_regex.match(age).group(1) # first matching group only extract the number
    return int(matching_number)

def generate_readmission_column(df, readmission_col_name):
    """
    """
    result_series = [0 if x == 'NO' else 1 for x in df[readmission_col_name]]
    return result_series




model_subset_columns = [           
                        ENCOUNTER_ID, 
                        RACE, 
                        GENDER, 
                        NUM_LABS, 
                        'num_procedures', 
                        'num_medications', 
                        'number_outpatient', 
                        'number_emergency', 
                        'number_inpatient', 
                        'number_diagnoses', 
                        'time_in_hospital'
                        ]


logging.info("Reading in data")
diabetes_df = pd.read_csv(DATA_PATH)

logging.info("Creating Model Matrix")
diabetes_df['age'].apply(extract_minimum_age)
model_subset = diabetes_df.loc[:, model_subset_columns].copy()
id_subset = diabetes_df.loc[:, [ENCOUNTER_ID, 'admission_type_id', 'discharge_disposition_id', 'admission_source_id']].copy()
ccs_subset = diabetes_df.loc[:, [ENCOUNTER_ID, 'CCS Category Description 1', 'CCS Category Description 2', 'CCS Category Description 3']].copy()
model_subset = pd.get_dummies(model_subset, prefix = "ind_", dummy_na = True, drop_first = True)
id_subset = pd.get_dummies(id_subset, columns = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id'],prefix = {x:x for x in id_subset.columns if x != 'encounter_id'}, dummy_na = True, drop_first = True)
ccs_subset = pd.get_dummies(ccs_subset, prefix = "ind_", dummy_na = True, drop_first = True)
ccs_subset = ccs_subset.groupby(ccs_subset.columns, axis = 1).sum()
model_dataset = (model_subset.merge(id_subset, how = "left", on = ENCOUNTER_ID)
                             .merge(ccs_subset, how = "left", on = ENCOUNTER_ID))

logging.debug(f"Model matrix has {model_dataset.shape[0]} rows")

model_dataset[READMISSION] = generate_readmission_column(diabetes_df, 'readmitted')


with open("./model_columns.json", "r") as f:
    model_cols = json.load(f)

model_dataset = model_dataset.reindex(model_cols, axis=1)
model_dataset = model_dataset.fillna(0)

logging.info("Creating train test splits")
X_train, X_test, y_train, y_test = (train_test_split(model_dataset
                                                     .drop(['encounter_id', 'readmission'], axis=1),
                                                     model_dataset['readmission'], 
                                                     test_size=0.2))

if args.model == 'rf':
    logging.info("Training model")
    model_ = RandomForestClassifier()
    model_.fit(X_train, y_train)

elif args.model == 'lr':
    logging.info("Training model")
    model_ = LogisticRegression()
    model_.fit(X_train, y_train)

logging.info("Generating predictions")
prediction_probabilities = model_.predict_proba(X_test)[:, 1]

logging.info(roc_auc_score(y_test, prediction_probabilities))
logging.info(average_precision_score(y_test, prediction_probabilities))