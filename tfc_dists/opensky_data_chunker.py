from datetime import date, timedelta
from traffic.data import opensky
import numpy as np
import joblib as jl
import os
from pathlib import Path
from glob import glob
import pandas as pd

DATA_DIR = 'data/southeng'
bounds = (-2.9, 50.5, 1.5, 51.9)
area_name = 'southeng'


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def clean_day(start, end):
    print(f'Cleaning {start}')
    if Path(f'{DATA_DIR}/{area_name}_tfc_clean_lt5000ft_{start}.pkl.bz2').exists():
        print(f'{start} already cached')
        return
    tfc = opensky.history(
        f"{start} 00:00",
        f"{end} 00:00",
        bounds=bounds,
        other_params=' and geoaltitude<=1524 and onground=false ')
    print(f'Downloaded data for {start}')
    tfc_clean = tfc.clean_invalid().assign_id().filter().resample('30s').eval(desc=None, max_workers=3)
    tfc_clean.data[tfc_clean.data.select_dtypes(np.float64).columns] = tfc_clean.data.select_dtypes(
        np.float64).astype(np.float16)
    tfc_clean.to_pickle(f'{DATA_DIR}/{area_name}_tfc_clean_lt5000ft_{start}.pkl.bz2', compression='bz2')


if __name__ == '__main__':
    start_date = date(2019, 1, 1)
    end_date = date(2020, 1, 2)
    date_strs = []
    for single_date in daterange(start_date, end_date):
        date_strs.append(single_date.strftime("%Y-%m-%d"))

    date_pairs = zip(date_strs, date_strs[1:])

    jl.Parallel(n_jobs=6)(jl.delayed(clean_day)(start, end) for start, end in date_pairs)

    month_dfs = []

    for m in range(1, 13):
        if Path(f'{area_name}_tfc_clean_lt5000ft_2019-{m:02d}.pkl.bz2').exists():
            continue
        month_pkls = glob(f'{DATA_DIR}/{area_name}_tfc_clean_lt5000ft_2019-{m:02d}-**.pkl.bz2')
        dfs = [pd.read_pickle(f) for f in month_pkls]
        month_df = pd.concat(dfs, axis=0, ignore_index=True)
        month_df.to_pickle(f'{DATA_DIR}/{area_name}_tfc_clean_lt5000ft_2019-{m:02d}.pkl.bz2', compression='bz2')
        month_dfs.append(month_df)

    year_df = pd.concat(month_dfs, axis=0, ignore_index=True)
    year_df.to_pickle(f'data/{area_name}_tfc_clean_lt5000ft_2019.pkl.bz2', compression='bz2')
