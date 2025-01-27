import polars as pl


def load_dataset(dataset: str = "diamonds") -> pl.DataFrame:
    """Load a dataset from the inria-soda tabular benchmark.

    Args:
        dataset: Name of the dataset to load. Defaults to "diamonds".

    Returns:
        pl.DataFrame: The loaded dataset
    """

    # path = fetch_file(
    #     f"https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_cat/{dataset}.csv",
    #     folder="data/raw",
    #     local_filename=f"{dataset}.csv",
    # )

    return pl.read_csv("../data/raw/diamonds.csv")
