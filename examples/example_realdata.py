# -*- coding: utf-8 -*-

"""
Example showing how to import a GPS Trajectory data file, and generate a tide acceleration correction
using the longmantide.solve_tide_df function.
"""

import sys
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from longmantide import import_trajectory, solve_tide_df


def run():
    file = '../tests/test_gps_data.txt'
    df = import_trajectory(file, timeformat='hms')
    corrected = solve_tide_df(df, lat='lat', lon='long', alt='ell_ht')

    date_fmt = DateFormatter("%H:%M:%S")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_ylabel('Total Correction [mGal]')
    ax.set_xlabel('Time')
    ax.xaxis.set_major_formatter(date_fmt)
    ax.plot_date(corrected.index, corrected['total_corr'], '-k', linewidth=1, xdate=True)
    plt.show()
    return 0


if __name__ == '__main__':
    sys.exit(run())
