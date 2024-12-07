import polars as pl

def fill_olumns_with_mean_by_city(data, group_column: str, columns_to_process: list[str]):
    filled_df = data
    for column in columns_to_process:
        city_medians = filled_df.group_by(group_column).agg(pl.col(column).mean().alias('median_'+column))
        city_median_map = dict(zip(city_medians[group_column], city_medians['median_'+column]))
        print(city_median_map)
        filled_df = filled_df.with_columns(
            pl.col(column).fill_null(
                pl.when(pl.col(group_column).is_not_null())
                .then(pl.col(group_column).replace_strict(city_median_map, default=0.0))
                .otherwise(0.0)
            )
        )

    return filled_df

