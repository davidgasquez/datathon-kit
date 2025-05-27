import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

data_url = "https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/"
train_data: pd.DataFrame = TabularDataset(f"{data_url}train.csv")
train_data.head()

label = "signature"
train_data[label].describe()


predictor = TabularPredictor(label=label).fit(
    train_data,
    ag_args_fit={"num_gpus": 1},
)

test_data = TabularDataset(f"{data_url}test.csv")
y_pred = predictor.predict(test_data.drop(columns=[label]))
print(y_pred[:5])
