import polars as pl


# TODO: Smoothing, quantiles, CV
def target_encoding(df: pl.DataFrame, target: str, column: str) -> pl.DataFrame:
    """Perform target encoding on a categorical column.

    Computes various statistics of the target variable for each category in the specified column.
    This is useful for converting categorical variables into numerical features while incorporating
    information from the target variable.

    Args:
        df: Input DataFrame
        target: Name of the target column to encode against
        column: Name of the categorical column to encode

    Returns:
        DataFrame with new columns containing the encoded features:
            - {column}_target_mean: Mean of target for each category
            - {column}_target_min: Min of target for each category
            - {column}_target_max: Max of target for each category
            - {column}_target_std: Standard deviation of target for each category
            - {column}_target_count: Count of occurrences for each category

    Example:
        >>> df = pl.DataFrame({
        ...     "category": ["A", "A", "B", "B", "C"],
        ...     "target": [1, 2, 3, 4, 5]
        ... })
        >>> target_encoding(df, "target", "category")
    """
    return df.with_columns(
        [
            pl.col(target).mean().over(pl.col(column)).alias(f"{column}_target_mean"),
            pl.col(target).min().over(pl.col(column)).alias(f"{column}_target_min"),
            pl.col(target).max().over(pl.col(column)).alias(f"{column}_target_max"),
            pl.col(target).std().over(pl.col(column)).alias(f"{column}_target_std"),
            pl.col(column).count().over(pl.col(column)).alias(f"{column}_target_count"),
        ]
    )
