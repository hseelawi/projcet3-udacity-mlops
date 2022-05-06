# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import dump
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, slice_performance
import json
# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv('../data/census_cleaned.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label='salary', training=False,
			encoder=encoder, lb=lb)

# Train and save a model.
model = train_model(X_train, y_train)

dump(model, '../model/classifier.joblib')
dump(encoder, '../model/encoder.joblib')
dump(lb, '../model/binariser.joblib')


# Predict test split labels
predicted_labels = inference(model, X_test)


# Overall performance
overall_performance = {}
pr, re, fb = compute_model_metrics(y_test, predicted_labels)
overall_performance['precision'] = pr
overall_performance['recall'] = re
overall_performance['fbeta'] = fb
with open('Metrics.json', 'w') as outfile:
    json.dump(overall_performance, outfile)

# Slices performance prediction
test['label'] = y_test
test['prediction'] = predicted_labels

performance_by_feature = {}
for cat in cat_features:
    performance_by_feature[cat] = slice_performance(test, cat)

with open('slice_output.txt', 'w') as outfile:
    json.dump(performance_by_feature, outfile)
