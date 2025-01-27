import polars as pl


def target_encoding(df: pl.DataFrame, target: str, column: str) -> pl.DataFrame:
    """Target encoding for a column."""

    return df.with_columns(
        pl.col(target).mean().over(pl.col(column)).alias(f"{column}_target_encoded")
    )
