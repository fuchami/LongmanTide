# -*- coding: utf-8 -*-

import sys
from datetime import datetime, timedelta

import pandas as pd
from matplotlib import pyplot as plt

from longmantide import solve_tide_df


def convert_matlab_serial_time(time: float):
    days = datetime.fromordinal(int(time))
    frac = timedelta(days=time % 1) - timedelta(days=366)
    return days + frac


def load_sample_dataset(fpath: str):
    df = pd.read_csv(fpath, header=0, names=['lon', 'lat', 'ser_time', 'total_corr'])

    dt_index = df['ser_time'].apply(convert_matlab_serial_time)
    dt_index.name = 'time'
    df.index = dt_index
    df['alt'] = 0
    df = df.drop(['ser_time'], axis=1)  # Drop the original ser_time column now that new index is set

    return df


def plot_time_correction(df, title='', y='total_corr'):
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    ax.set_ylabel('Total Correction [mGal]')
    ax.set_xlabel('Date/Time')

    ax.plot_date(df.index, df[y], '-k', linewidth=1, xdate=True)
    plt.show()


if __name__ == '__main__':
    sample_ds = load_sample_dataset('../tests/matlab_synthetic.csv')
    # plot_time_correction(sample_ds, title='Matlab Synthetic Correction')

    corrected = solve_tide_df(sample_ds, lat='lat', lon='lon', alt='alt')

    print(sample_ds.iloc[7])
    print(corrected.iloc[7])

    # plot_time_correction(corrected, title='Longman tide calculated')

    delta = corrected['total_corr'] - sample_ds['total_corr']
    print("Delta (corrected - sample)")
    print(delta)
    print(delta.describe())

    sys.exit(0)
