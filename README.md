# Longman Tide

Longman Tide is a Python library which implements the scheme for computing the tidal accelerations due to the moon and
sun, as published by I.M. Longman in the Journal of Geophysical Research, Vol 64, no. 12, 1959

### Requirements

The numpy and Pandas libraries are required for processing tide corrections, and importing trajectory data for correction

The matplotlib library is currently only used in the examples to give a visual representation of the data.


### References

I.M. Longman "Forumlas for Computing the Tidal Accelerations Due to the Moon
and the Sun" Journal of Geophysical Research, vol. 64, no. 12, 1959,
pp. 2351-2355

P. Schureman "Manual of harmonic analysis and prediction of tides" U.S. Coast and Geodetic Survey, 1958


#### Acknowledgements

This library is based on the work of John Leeman's LongmanTide Python implementation.



### Examples

There are several example scripts in the examples directory illustrating how to use the longmantide solving functions.

Here is a simple demonstration of calculating a correction series for a static latitude/longitude/altitude over a
specified time period, with intervals of 1 second.

```python

from datetime import datetime
from longmantide import solve_point_corr

# Example static data for Denver, January 1, 2018
lat = 39.7392
lon = -104.9903
alt = 1609.3
t0 = datetime(2018, 1, 1, 12, 0, 0)

# Calculate corrections for one day (60*60*24 points), with 1 second resolution
result_df = solve_point_corr(lat, lon, alt, t0, n=60*60*24, increment='S')

# Result is a pandas DataFrame, with a DatetimeIndex, and correction values in the 'total_corr' column
# i.e.
corrections = result_df['total_corr'].values  # get the internal Numpy ndarray for corrections

# Plot the corrections using matplotlib
from matplotlib import pyplot as plt

plt.plot(corrections)
plt.ylabel('Tidal Correction [mGals]')
plt.show()

```
