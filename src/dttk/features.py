import polars as pl


def target_encoding(
    df: pl.DataFrame,
    target: str,
    column: str,
    alpha: float = 0.0,
    stats: list[str] = ["mean", "min", "max", "std", "count", "q25", "q50", "q75"],
) -> pl.DataFrame:
    """Perform target encoding on a categorical column with smoothing support.

    Args:
        df: Input DataFrame
        target: Target column to encode against
        column: Categorical column to encode
        alpha: Smoothing parameter (0=no smoothing, 1=full smoothing)
        stats: List of statistics to compute (supported: mean, min, max, std, count, unique, q25, q50, q75)

    Returns:
        DataFrame with encoded features.

    Example:
        >>> df = pl.DataFrame({
        ...     "category": ["A", "A", "B", "B", "C"],
        ...     "target": [1, 2, 3, 4, 5]
        ... })
        >>> target_encoding(df, "target", "category", alpha=0.1)
    """

    supported_stats = {
        "mean": pl.col(target).mean(),
        "min": pl.col(target).min(),
        "max": pl.col(target).max(),
        "std": pl.col(target).std(),
        "count": pl.col(target).count(),
        "unique": pl.col(target).n_unique(),
        "q25": pl.col(target).quantile(0.25),
        "q50": pl.col(target).quantile(0.50),
        "q75": pl.col(target).quantile(0.75),
    }

    if invalid := [s for s in stats if s not in supported_stats]:
        raise ValueError(
            f"Invalid stats {invalid}. Supported: {list(supported_stats.keys())}"
        )

    if not (0 <= alpha <= 1):
        raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

    # Build aggregations
    aggregations = [
        supported_stats[stat].over(column).alias(f"{column}_{target}_{stat}")
        for stat in stats
    ]

    # Compute base statistics
    df = df.with_columns(aggregations)

    # Apply smoothing to mean if requested
    if alpha > 0:
        if "mean" not in stats:
            raise ValueError("Smoothing requires 'mean' to be in stats")

        global_mean = df.select(pl.mean(target)).item()
        smoothed_mean = (
            pl.col(f"{column}_{target}_mean") * (1 - alpha) + global_mean * alpha
        )

        df = df.with_columns(smoothed_mean.alias(f"{column}_{target}_mean"))

    return df
