import json
import os

import pandas as pd
import geopandas as gpd
import numpy as np
from uuid import uuid4
import datetime as dt
import subprocess
from glob import glob
import joblib as jl


def log2czml(logfile):
    outfile = open(logfile.split('.')[0] + '.json', 'w')
    subprocess.call(['python', 'mavlogdump.py', '--format=json', logfile], stdout=outfile, shell=True)
    with open(outfile.name, 'r') as f:
        raw_str = f.readlines()
    pos_str = [json.loads(line) for line in raw_str if 'POS' in line]
    pos_df = pd.json_normalize(pos_str)

    pos_df = pos_df[pos_df['meta.type'] == 'POS'].dropna(axis=1).drop_duplicates(
        subset=['data.Lat', 'data.Lng', 'data.Alt']).drop(
        labels=['meta.type', 'data.TimeUS', 'data.RelHomeAlt', 'data.RelOriginAlt'], axis=1).iloc[::20, :]
    pos_gdf = gpd.GeoDataFrame(pos_df,
                               geometry=gpd.points_from_xy(pos_df['data.Lng'], pos_df['data.Lat'], crs='epsg:4326'))

    pos_df = pos_gdf[['meta.timestamp', 'data.Lng', 'data.Lat', 'data.Alt']]
    flat_4d_coords = list(np.array(pos_gdf[['meta.timestamp', 'data.Lng', 'data.Lat', 'data.Alt']].values).flatten())
    flat_3d_coords = list(np.array(pos_gdf[['data.Lng', 'data.Lat', 'data.Alt']].values).flatten())
    max_time = dt.datetime.fromtimestamp(pos_df['meta.timestamp'].max())
    min_time = dt.datetime.fromtimestamp(pos_df['meta.timestamp'].min())
    uid = str(uuid4())

    czml_id_packet = {
        'id': "document",
        'name': "CZML Model",
        'version': "1.0",
    }
    czml_model_packet = {
        'id': uid + '.log',
        'availability': min_time.replace(microsecond=0).isoformat() + 'Z/' + max_time.replace(
            microsecond=0).isoformat() + 'Z',
        'position': {"epoch": "1970-01-01T00:00:00Z",
                     "cartographicDegrees": flat_4d_coords},
        'model': {
            "gltf": "https://raw.githubusercontent.com/bobbyhiom/cyberpunkjam/master/CyberPunkScene/models/drone.gltf",
            "scale": 1, "minimumPixelSize": 32}
    }
    czml_traj_packet = {
        'id': uid + '.log',
        'availability': min_time.replace(microsecond=0).isoformat() + 'Z/' + max_time.replace(
            microsecond=0).isoformat() + 'Z',
        'polyline': {"positions": {
            "cartographicDegrees": flat_3d_coords
        },
            "material": {"solidColor": {"color": {"rgba": [200, 255, 200, 255]}}}}
    }
    with open(f'{outfile.name}.czml', 'w') as f:
        # f.write(str(czml_id_packet))
        # f.write(',')
        # f.write(str(czml_model_packet))
        # f.write(',')
        f.write(str(czml_traj_packet))


if __name__ == '__main__':
    logfiles = glob('data/**/*.bin', recursive=True)

    jl.Parallel(n_jobs=-1)(jl.delayed(log2czml)(logfile) for logfile in logfiles)
