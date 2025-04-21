import polars as pl
import skrub
import skrub.datasets


def load_credit_fraud_dataset() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return tables from Skrub's credit fraud dataset.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame]: The Polars tables from the Skrub dataset.
    """

    baskets = pl.from_pandas(
        skrub.datasets.fetch_credit_fraud(data_home="/tmp").baskets
    )
    products = pl.from_pandas(
        skrub.datasets.fetch_credit_fraud(data_home="/tmp").products
    )

    return baskets, products
