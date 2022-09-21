import datetime
from datetime import timedelta, date
from pathlib import Path
from typing import Iterable, List, Generator
import pandas as pd
import geopandas as gpd
import numpy as np
from ratelimit import sleep_and_retry, limits

BASE_ENDPOINT = 'https://bgaladder.net/API/'
MAX_REQ_PER_MIN = 15  # Total call limit per minute
RL_PERIOD = 20  # Period over which the above will be calculated. "Spreads out" the calls over the course of a minute
MORNING_CUTOFF_HOUR = 6  # Hour of day at which to stop execution
DATA_DIR = 'data/bgaladder/'

assert RL_PERIOD < 61


def make_url(*args: Iterable[str]) -> str:
    return BASE_ENDPOINT + '/'.join([*args])


def daterange(start_date: date, end_date: date) -> Generator[date, None, None]:
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


@sleep_and_retry
@limits(calls=MAX_REQ_PER_MIN // (60 // RL_PERIOD), period=RL_PERIOD)
def get(*args, **kwargs):
    """
    A single method for calls, so we can rate limit requests to the server as a whole rather than per endpoint
    """
    import requests
    return requests.get(*args, **kwargs)


def get_flight_ids(query_date: date) -> List[int]:
    return get(make_url('FlightIDs', query_date.strftime('%d-%b-%Y'))).json()


def get_trace(flight_id: int) -> gpd.GeoDataFrame:
    flight_log = get(make_url('FlightLog', str(flight_id))).json()
    return gpd.GeoDataFrame(flight_log)


def clean_to_traffic_df(raw_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean and munge df to format expected by traffic library
    """
    if not raw_df.shape[0]:
        return raw_df
    raw_df['squawk'] = None
    raw_df['callsign'] = None
    raw_df['onground'] = False
    raw_df['spi'] = False
    raw_df['alert'] = False
    out_df = raw_df.rename(
        {'Latitude': 'latitude', 'Longitude': 'longitude', 'Altitude': 'altitude', 'Time': 'timestamp',
         'RateOfClimb': 'vertical_rate'}, axis=1)
    out_df = out_df.drop(labels=['ENL'], axis=1, errors='ignore')
    out_df['timestamp'] = pd.to_datetime(out_df['timestamp'])
    out_df['icao24'] = out_df['flight_id']
    out_df['callsign'] = out_df['flight_id']
    out_df['altitude'] = out_df['altitude'].astype(float)
    out_df[out_df.select_dtypes(np.float64).columns] = out_df.select_dtypes(np.float64).astype(np.float16)
    out_df[out_df.select_dtypes(np.int64).columns] = out_df.select_dtypes(np.int64).astype(np.int16)
    return out_df


def is_early_morning():
    return datetime.datetime.now().hour < MORNING_CUTOFF_HOUR


if __name__ == '__main__':
    start_date = date(2019, 1, 1)
    end_date = date(2020, 1, 2)

    assert end_date > start_date

    trace_gdfs = []

    for d in daterange(start_date, end_date):
        if not is_early_morning():
            # Stop execution if past morning cutoff to reduce server load
            exit()
        if not Path(f'{DATA_DIR}/daily/bgaladder_raw_{d.year}-{d.month:02d}-{d.day:02d}.pkl.bz2').exists():
            day_ids = get_flight_ids(d)
            if not day_ids:
                continue
            day_dfs = []
            print(f'Downloading traces for {d}')
            for fid in day_ids:
                flight_df = get_trace(fid)
                flight_df['flight_id'] = fid
                if flight_df.shape[0]:
                    day_dfs.append(flight_df)
            day_df = clean_to_traffic_df(pd.concat(day_dfs, axis=0))
            day_df.to_pickle(f'{DATA_DIR}/daily/bgaladder_raw_{d.year}-{d.month:02d}-{d.day:02d}.pkl.bz2',
                             compression='bz2')
        else:
            day_df = pd.read_pickle(f'{DATA_DIR}/daily/bgaladder_raw_{d.year}-{d.month:02d}-{d.day:02d}.pkl.bz2')
        trace_gdfs.append(day_df)

    total_gdf = pd.concat(trace_gdfs, axis=0)
    pd.read_pickle(f'{DATA_DIR}/bgaladder_raw_2019.pkl.bz2')
