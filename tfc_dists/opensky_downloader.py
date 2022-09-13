import traffic
from traffic.data import opensky
import numpy as np

bounds = (-2.9, 50.5, 1.5, 51.9)
area_name = 'southeng'


if __name__ == '__main__':
    tfc = opensky.history(
        "2019-01-01 00:00",
        "2020-01-01 00:00",
        bounds=bounds,
        other_params=' and geoaltitude<=1524 and onground=false ')
    tfc.to_pickle(f"../data/{area_name}_tfc_raw_lt5000ft_2019.pkl")
    tfc_clean = tfc.clean_invalid().assign_id().filter().resample('30s').eval(desc='', max_workers=-1)
    tfc_clean.data[tfc_clean.data.select_dtypes(np.float64).columns] = tfc_clean.data.select_dtypes(np.float64).astype(np.float16)
    tfc_clean.to_pickle(f"../data/{area_name}_tfc_clean_lt5000ft_2019.pkl")