import optuna
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import load_dataset

pl.Config.set_tbl_rows(5)
pl.Config.set_tbl_cols(20)

# Load datasets
train_df = load_dataset("diamonds")
test_df = load_dataset("diamonds")

# Load external datasets
country_df = pl.read_csv("data/external/country_statistics.csv")

# # Merge datasets
train_df = train_df.join(country_df, on="country", how="left")
test_df = test_df.join(country_df, on="country", how="left")

# Define transformers for different column types
numeric_features = ["carat", "depth", "table", "x", "y", "z"]
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_features = ["cut", "color", "clarity"]
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


def objective(trial):
    # Define hyperparameter search space
    params = {
        "model__n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "model__max_depth": trial.suggest_int("max_depth", 3, 20),
        "model__min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "model__random_state": 42,
    }

    # Create pipeline with trial parameters
    pipeline = Pipeline(
        steps=[
            ("transform", preprocessor),
            ("model", RandomForestRegressor()),
        ]
    )
    pipeline.set_params(**params)

    # Perform cross-validation
    cv_scores = cross_val_score(
        pipeline,
        train_df,  # type: ignore
        train_df["price"],
        cv=5,
        scoring="r2",
    )
    return cv_scores.mean()


# Create and run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print optimization results
print("Best trial:")
print(f"  Value: {study.best_trial.value:.3f}")
print("  Params:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# Create final pipeline with best parameters
final_pipeline = Pipeline(
    steps=[
        ("transform", preprocessor),
        ("model", RandomForestRegressor(**study.best_trial.params, random_state=42)),
    ]
)

# Train final model
final_pipeline.fit(train_df, train_df["price"])
