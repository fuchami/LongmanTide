Tide Gravity
============

tidegravity is a Python library which implements Ivor Longman's scheme for computing the tidal accelerations due to the
moon and sun, as published by I.M. Longman in the Journal of Geophysical Research, Vol 64, no. 12, 1959
This can be useful for correcting gravimetric survey data, as the gravitational forces due to the tidal effects of the
Sun and Moon can be on the order of several hundred microGals, depending on the surveyors location and the relative
positions of the Sun and Moon to each other, and the surveyor.

Requirements
------------

- numpy
- pandas

The numpy and pandas libraries are required for processing tide corrections, and importing trajectory data for correction

The matplotlib library is an optional requirement and is currently only used in the examples to plot a visual
representation of the data.

API
---

.. role:: py(code)
    :language: python

The following API functions are provided (subject to change in future releases):

* :py:`solve_longman_tide(lat, lon, alt, time)`

  Solve for total gravity correction due to Sun/Moon from numpy array inputs
* :py:`solve_longman_tide_scalar(lat, lon, alt, time)`

  Wrapper around solve_longman_tide, accepts single scalar values for lat/lon/alt and a single DateTime object
* :py:`solve_point_corr(lat, lon, alt, t0, n=3600, increment='S')`

  Return tidal correction over a time span defined by t0 with n points at given increment for static (scalar)
  position parameters
* :py:`solve_tide_df(df, lat='lat', lon='lon', alt='alt')`

  Wrapper accepting a pandas DataFrame (df) object as the input, df should have a DatetimeIndex, and lat/lon/alt
  columns. Alternate column names can be provided via parameters, which will then be used to extract components from
  the input DataFrame.


References
----------

* I.M. Longman "Forumlas for Computing the Tidal Accelerations Due to the Moon
  and the Sun" Journal of Geophysical Research, vol. 64, no. 12, 1959,
  pp. 2351-2355
* P\. Schureman "Manual of harmonic analysis and prediction of tides" U.S. Coast and Geodetic Survey, 1958


Acknowledgements
----------------

.. _LongmanTide: https://github.com/jrleeman/LongmanTide

This library is based on the work of John Leeman's LongmanTide Python implementation.
LongmanTide_


Examples
--------

There are several example scripts in the examples directory illustrating how to use the longmantide solving functions.

Here is a simple demonstration of calculating a correction series for a static latitude/longitude/altitude over a
specified time period, with intervals of 1 second.

.. code-block:: python

    from datetime import datetime
    from tidegravity import solve_point_corr

    # Example static data for Denver, January 1, 2018
    lat = 39.7392
    lon = -104.9903
    # Note: West should be entered as a negative longitude value
    alt = 1609.3
    t0 = datetime(2018, 1, 1, 12, 0, 0)

    # Calculate corrections for one day (60*60*24 points), with 1 second resolution
    result_df = solve_point_corr(lat, lon, alt, t0, n=60*60*24, increment='S')

    # Result is a pandas DataFrame, with a DatetimeIndex, and correction
    # values in the 'total_corr' column i.e.
    corrections = result_df['total_corr'].values

    # Plot the corrections using matplotlib
    from matplotlib import pyplot as plt

    plt.plot(corrections)
    plt.ylabel('Tidal Correction [mGals]')
    plt.show()

