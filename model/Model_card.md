# Model Card

## Model Details

This model constitutes a Random Forest classifier trained on the [census income dataset](https://archive.ics.uci.edu/ml/datasets/census+income). It was trained on the 1st of May, 2022, by Haitham Seelawi. The random classifier uses the sklearn implementation with the default hyperparameters.

## Intended Use
This model is intended to predict a persons outcome (more than 50k dollars a year or less), based on a set of personal attributes.

## Training and Evaluation data
This data was extraced from the 1994 Census database by Barry Becker. It was cleaned based on a set of resonable data cleaning rules
This data was split into 80/20 training/evaluation splits using. The full dataset contains 48,842 examples and 14 attributes (numerical + categorical)

## Metrics
We use three metics namely, precision, recall, and f1 score. The model achieves 0.75, 0.6279, and 0.6835 respectively.

## Ethical Considerations
While the model was tested for performance on slices of data, it was not full vetted for any form of under-representation or other factors that might make it not biased against certain ethnicities.

## Caveats and Recommendations
The data the model trained on is very old (1994), which indicates that the model will be subject to a severe concept and data drift.
