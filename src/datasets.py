import polars as pl


def load_dataset() -> pl.DataFrame:
    return pl.read_csv(
        "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_cat/diamonds.csv"
    )
