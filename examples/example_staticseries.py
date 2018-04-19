# -*- coding: utf-8 -*-

"""
Example showing how to generate a DataFrame of tide corrections given a static lat/long/altitude
over a specified period, from a start date/time t0
"""

import sys
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from longmantide import solve_point_corr


def run():
    # Example Data (Denver, Jan 1, 2018)
    lat = 39.7392
    lon = -104.9903
    alt = 1609.3
    t0 = datetime(2018, 1, 1, 12, 0, 0)

    # Calculate minute resolution data over a one week period
    result = solve_point_corr(lat, lon, alt, t0, n=60*24*28, increment='min')
    print(result.describe())
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    date_fmt = DateFormatter("%m-%d")
    ax.set_ylabel('Total Correction [mGal]')
    ax.set_xlabel('Date [month-day]')
    ax.xaxis.set_major_formatter(date_fmt)
    ax.plot_date(result.index, result.total_corr, '-k', linewidth=1, xdate=True)
    plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(run())


