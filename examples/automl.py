import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

# Load data
data_url = "https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/"
train_data: pd.DataFrame = TabularDataset(f"{data_url}train.csv")

# Train model
label = "signature"
predictor = TabularPredictor(label=label).fit(train_data)

# Predict
test_data = TabularDataset(f"{data_url}test.csv")
y_pred = predictor.predict(test_data.drop(columns=[label]))
print(y_pred[:5])
