## Control experiment with one-hot encoding and no normalization.

import os

# Must import before sklearn or else ImportError.
from comet_ml import Experiment
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import yaml

# Custom function to load pre-split train, validation and test data as pandas
# DataFrames.
import sys

sys.path.append("../")
from data_processing.loading import load_train_valid_data, load_test_data

RANDOM_STATE = 285
MAX_FEATURES = 10000
MAX_ITER = 10000
INV_REGULARIZATION_STRENGTH = 1e5


# Function for logging prediction files to CometML.
def log_predictions(experiment, pred_labels, index, filename):
    preds_df = pd.DataFrame(pred_labels, index=index, columns=["Prediction"])

    # Adjust ID column name to fit the format expected in the output.
    preds_df.index.name = "Id"

    # Log predictions as a CSV to CometML, retrievable under the experiment
    # by going to `Assets > dataframes`.
    experiment.log_table(filename=filename, tabular_data=preds_df)

    # Log predictions to local file system
    preds_df.to_csv(filename)

def main():
    experiment = Experiment(
        api_key="",
        project_name="cil-project",
        workspace="smueksch",
        # Prevent CometML from dumping all sorts of parameter info about SVC,
        # embedding, etc. into the experiment log. Avoids cluttering but could
        # be preferrable to set to True.
        auto_param_logging=False,
        disabled=True,
    )
    experiment.set_name("Dict Normalization")

    # Define experiment parameters for CometML to be logged to the project under
    # the experiment.
    params = {
        "random_state": RANDOM_STATE,
        "embedding": "one-hot",
        "max_features": MAX_FEATURES,
        "max_iterations": MAX_ITER,
        "inv_regularization_strength": INV_REGULARIZATION_STRENGTH,
    }
    experiment.log_parameters(params)

    pipeline = Pipeline(
        [
            ("embedding", CountVectorizer(max_features=MAX_FEATURES)),
            (
                "classifier",
                LogisticRegression(
                    C=INV_REGULARIZATION_STRENGTH,
                    max_iter=MAX_ITER,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # Need to provide the relative path to the folder containing the
    # training_set.csv and validation_set.csv
    path_to_dataset = os.path.join("dataset-normalized")
    train, valid = load_train_valid_data(path_to_dataset)
    test = load_test_data(path_to_dataset)

    # Train
    pipeline.fit(train["tweet"], train["label"])
    train_pred = pipeline.predict(train["tweet"])
    train_score = accuracy_score(train["label"], train_pred)
    experiment.log_confusion_matrix(
        train["label"], train_pred, title="Confusion Matrix (Train)"
    )

    # Validation
    valid_pred = pipeline.predict(valid["tweet"])
    valid_score = accuracy_score(valid["label"], valid_pred)
    experiment.log_confusion_matrix(
        valid["label"], valid_pred, title="Confusion Matrix (Validation)"
    )
    log_predictions(experiment, valid_pred, valid.index, "valid_predictions.csv")

    # Test predictions
    predicted_labels = pipeline.predict(test["tweet"])
    log_predictions(experiment, predicted_labels, test.index, "test_predictions.csv")

    # Define experiment metrics for CometML to be logged to the project under
    # the experiment.
    metrics = {"train_score": float(train_score), "valid_score": float(valid_score)}
    experiment.log_metrics(metrics)
    # Log metrics to local file system
    with open('metrics.yaml', 'w+') as outfile:
        yaml.dump(metrics, outfile)


if "__main__" == __name__:
    main()
