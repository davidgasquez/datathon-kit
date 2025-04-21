import altair as alt
import polars as pl
from sklearn.inspection import permutation_importance


def compute_feature_importance(model, X, y, **kwargs):
    """Calculate the permutation importance score for a model.

    Args:
        model: The model to calculate the permutation importance score for.
        X: The input data.
        y: The target data.
        **kwargs: Additional keyword arguments to pass to the permutation_importance function.

    Returns:
        The permutation importance score for the model.
    """

    return permutation_importance(model, X, y, **kwargs)


def plot_feature_importance(importance, feature_names, top_n=10):
    """Plot the feature importance scores.

    Args:
        importance: The feature importance scores.
        feature_names: The feature names.
        top_n: The number of features to plot.

    Returns:
        The Altair chart.
    """

    # Create a DataFrame from the importance and feature names
    df = pl.DataFrame({"importance": importance, "feature": feature_names})

    # Sort the DataFrame by importance in descending order
    df = df.sort("importance", descending=True)

    # Select the top N features
    df = df.head(top_n)

    # Create an Altair bar chart
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="importance",
            y=alt.Y("feature", sort="-x"),  # Sort by importance descending
        )
        .properties(title="Feature Importance")
    )

    return chart
