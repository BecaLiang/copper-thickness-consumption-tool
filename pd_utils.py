from datetime import date, datetime

import numpy as np
import pandas as pd

from utils import weekdays, get_last_week_of_year
from stg_utils.country_info import CountryInfo


def get_week_table(
    first_year: int, 
    last_year: int=None, 
    start_week: int=None,
    day_of_week: int=None,
) -> pd.Series:
    # Future Feature, expand into past
    last_year = date.today().year if last_year is None else last_year
    no_of_weeks_in_year = {
        year: get_last_week_of_year(year) for year in range(
            int(first_year), 
            int(last_year+1),
            1,
        )
    }
    idcs = pd.MultiIndex.from_arrays(
        [
            np.concatenate(
                [
                    np.array(
                        [year] * weeks
                    ) for year, weeks in no_of_weeks_in_year.items()
                ]
            ),
            np.concatenate(
                [
                    np.arange(
                        1, weeks+1, 1
                    ) for year, weeks in no_of_weeks_in_year.items()
                ]
            ),
        ],
        names=['year', 'weeks'],
    )
    df = pd.DataFrame(
        index=idcs,
        data=np.ones(
            sum(no_of_weeks_in_year.values())
        ),
        columns=['tmp_one'],
    )
    df.loc[:, 'week_running'] = df.tmp_one.cumsum() - 1
    if start_week is not None:
        df.loc[:, 'week_running'] = df.week_running \
            - df.loc[first_year, start_week].week_running
    if day_of_week is not None:
        assert day_of_week in range(1, 8, 1)
        df.loc[:, weekdays[day_of_week]] = np.arange(
            datetime.fromisocalendar(
                int(first_year),
                1,
                day_of_week,
            ),
            datetime.fromisocalendar(
                int(last_year), 
                no_of_weeks_in_year[last_year], 
                day_of_week,
            ) + np.timedelta64(1, 'D'),
            np.timedelta64(7, 'D'),
            dtype=np.datetime64,
        )

    return df.drop('tmp_one', axis=1)


def join_weekday_gbweek_on_frame(dfin: pd.DataFrame, dt_column: str, **kwargs):
    dfwd = get_week_table(dfin[dt_column].dt.year.min(), **kwargs)
    week_col, year_col = f'_TMP_{dt_column}_week', f'_TMP_{dt_column}_year'
    assert week_col not in dfin.columns and year_col not in dfin.columns
    dfin.loc[:, year_col] = dfin[dt_column].dt.year
    dfin.loc[:, week_col] = dfin[dt_column].dt.isocalendar()['week']
    dfin = dfin.join(dfwd, on=[year_col, week_col])
    return dfin.drop([year_col, week_col], axis=1)


def apply_equal_statistics_binning(ps_input: pd.Series, bin_boundaries: np.array) -> pd.Series:
    return np.digitize(ps_input, bin_boundaries)


def fit_equal_statistics_binning(ps_input, nbins: int=100) -> tuple[pd.Series, np.array]:
    arr = ps_input.sort_values().values
    bin_boundaries = arr[::int(np.ceil(len(ps_input) / nbins))]
    return apply_equal_statistics_binning(ps_input, bin_boundaries), bin_boundaries


def add_equal_binning_to_df(
    dfin: pd.DataFrame, 
    column: str, 
    nbins: int=100, 
    new_column_name: str=None,
    return_boundaries: bool=True,
) -> tuple[pd.DataFrame, np.array] | pd.DataFrame:
    binned_arr, bin_boundaries = fit_equal_statistics_binning(
        ps_input=dfin[column], 
        nbins=nbins
    )
    dfin.loc[:, new_column_name or column + '_flat'] = binned_arr
    if return_boundaries:
        return dfin, bin_boundaries
    else:
        return dfin
    

def make_series_division_safe(
        ps: pd.Series,
        epsilon: float=1e-10,
        ):
    
    assert epsilon > 0

    ps_tmp = pd.Series(
        index=ps.index,
        data=np.zeros_like(ps),
        name=ps.name,
    )

    ps_small_abs = (ps > -epsilon) & (ps < epsilon)
    ps_tmp[ps_small_abs & (ps >= 0)] = epsilon
    ps_tmp[ps_small_abs & (ps < 0)] = -epsilon

    return ps + ps_tmp


def get_most_frequent_category(
        df: pd.DataFrame, 
        major_column: str, 
        count_column: str,
        transform: bool=False,
    ) -> pd.Series:
    df_tmp_1 = df[[major_column, count_column]].copy()
    df_tmp_1.loc[:, 'one'] = 1
    df_tmp_2 = df_tmp_1.groupby(
        [major_column, count_column]
        )['one'].sum().rename('occurences').reset_index()
    psmax = df_tmp_2.loc[
        df_tmp_2.groupby(major_column)['occurences'].idxmax()][
            [major_column, count_column]
        ].set_index(major_column)[
            count_column].rename(count_column)
    if not transform:
        return psmax
    else:
        return df_tmp_1[major_column].map(psmax).rename(count_column)
    

def add_continent_based_country(
        df: pd.DataFrame,
        country_column: str,
        new_column_name: str,
        plain_countries: list[str]=None,
        plain_continents: list[str]=None,
        file_name: str=None,
        index_column: str='two_letter_code',
        other_str: str='World',
        overwrite: bool=True,
        **kwargs,
):
    if plain_continents is None:
        plain_continents=['Europe']
    if plain_countries is None:
        plain_countries = ['DE', 'CN']
    ci = CountryInfo.init_from_csv(file_path=file_name, **kwargs)
    if new_column_name in df.columns:
        if overwrite:
            df = df.drop(new_column_name, axis=1)
        else:
            raise KeyError(f'{new_column_name} already part of the columns')
    df = df.join(
        ci.get(
            column_code='continent',
            index=index_column,
        ).rename(
            new_column_name,
        ),
        on=country_column,
    )
    df.loc[df[country_column].isin(plain_countries), new_column_name] = \
        df.where(df[country_column].isin(plain_countries))[country_column]
    df.loc[
        ~df[new_column_name].isin(plain_continents) & 
        ~df[country_column].isin(plain_countries),
        new_column_name
    ] = other_str
    return df
