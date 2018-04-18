# -*- coding: utf-8 -*-

import numpy as np
from datetime import datetime
from math import sin, radians


__all__ = ['calculate_julian_century', 'solve_longman_tide']


"""
Longman Earth Tide Calculator - Adopted from jrleeman's implementation of 
I. M. Longman's earth tide calculations (see references below) 

Parts of this program (c) 2017 John Leeman
Licensed under the MIT License


References
----------
I.M. Longman "Forumlas for Computing the Tidal Accelerations Due to the Moon 
and the Sun" Journal of Geophysical Research, vol. 64, no. 12, 1959, 
pp. 2351-2355

P. Schureman "Manual of harmonic analysis and prediction of tides" U.S. Coast 
and Geodetic Survey, 1958

John Leeman's GitHub page for the original implementation: 
https://github.com/jrleeman/LongmanTide


ToDo
----
1. Return corrections indexed by datetime (possibly use pandas)
2. Allow input data as a dataframe with named columns, and/or way to map 
columns to data
3. Test against Matlab implementation (consider altitutde - I don't think the 
Matlab implementation factors this in)

"""

# Constants #
# See corresponding definitions in Longman 1959
μ = 6.673e-8  # Newton's gravitational constant in cgs units (Verify this,
# should be 6.674e-8?)
M = 7.3537e25  # Mass of the moon in grams
S = 1.993e33  # Mass of the sun in grams
e = 0.05490  # Eccentricity of the moon's orbit
m = 0.074804  # Ratio of mean motion of the sun to that of the moon
c = 3.84402e10  # Mean distance between the centers of the earth and the moon in cm
c1 = 1.495e13  # Mean distance between centers of the earth and sun in cm
h2 = 0.612  # Love parameter  # See: https://en.wikipedia.org/wiki/Love_number
k2 = 0.303  # Love parameter  # See Also:
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4599447/
love = 1 + h2 - 1.5 * k2
a = 6.378270e8  # Earth's equitorial radius in cm
i = 0.08979719  # (i) Inclination of the moon's orbit to the ecliptic
ω = radians(23.452)  # Inclination of the Earth's equator to the ecliptic
origin_date = datetime(1899, 12, 31, 12, 00, 00)  # Noon Dec 31, 1899
# End constants declaration #


def calculate_julian_century(dates: np.ndarray):
    """
    Same as calculate_julian_century but operates on numpy array objects,
    and returns numpy arrays of century/hour values.

    Parameters
    ----------
    dates : np.ndarray
        1-dimensional array of DateTime objects to convert into
        century/hours arrays

    Returns
    -------
    2-tuple of:
        T : np.ndarray
            Number of Julian centuries (36525 days) from GMT Noon on
            December 31, 1899
        t0 : np.ndarray
            Greenwich civil dates measured in hours

    """
    delta = dates - origin_date
    days = np.array([x.days + x.seconds / 3600. / 24. for x in delta])
    t0 = np.array([x.hour + x.minute / 60. + x.second / 3600. for x in
                   dates])
    return days / 36525, t0


def solve_longman_tide(lat: np.ndarray, lon: np.ndarray,
                       alt: np.ndarray, time: np.ndarray):
    """
    Find the total gravity correction due to the Sun/Moon for the given
    latitude, longitude, altitude, and time - supplied as numpy 1-d arrays.
    Corrections are calculated for each corresponding set of data in the
    supplied arrays, all arrays must be of the same shape (length and dimension)

    This function returns the Lunar, Solar, and Total gravity corrections as
    a 3-tuple of numpy 1-d arrays.

    Parameters
    ----------
    lat : np.ndarray
        1-dimensional array of float values denoting latitudes
    lon : np.ndarray
        1-dimensional array of float values denoting longitudes
    alt : np.ndarray
        1-dimensional array of float values denoting altitude in meters
    time : np.ndarray
        1-dimensional array of DateTime objects denoting the time series

    Returns
    -------
    3-Tuple
        gMoon: Vertical component of tidal acceleration due to the moon
        gSun: Vertical component of tidal acceleration due to the sun
        gTotal: Total vertical component of tidal acceleration (moon + sun)

    """

    assert lat.shape == lon.shape == alt.shape == time.shape

    T, t0 = calculate_julian_century(time)
    T2 = T ** 2
    T3 = T ** 3

    t0[t0 < 0] += 24
    t0[t0 >= 24] -= 24

    # lat/lon is defined with West positive (+) in the Longman paper
    lon = -1 * lon
    λ = np.radians(lat)  # λ Latitude of point P
    cosλ = np.cos(λ)
    sinλ = np.sin(λ)

    ht = alt * 100  # height above sea-level of point P in centimeters (cm)

    #
    # Lunar Calculations #
    #

    # s Mean longitude of moon in its orbit reckoned from the referred
    # equinox
    # TODO: What is 1336 rev. ?
    # Constants from Bartels [1957 pp. 747] eq (10')
    # 270°26'11.72" + (1336 rev. + 1,108,411.20")T + 7.128" * T2 + 0.0072" * T3
    # Converting degrees/minutes/seconds to radians gives us the constants
    # TODO: use constants from Schureman or Bartels 1957 (coeff of T2 and T3)
    s = 4.72000889397 + 8399.70927456 * T + 3.45575191895e-05 * \
        T2 + 3.49065850399e-08 * T3
    # p Mean longitude of lunar perigee
    # constants from Bartels [1957] eq (11')
    # p = 334° 19' 46.42" + (11 rev. + 392,522.51") T - 37.15" * T2 - 0.036" T3
    p = 5.83515162814 + 71.0180412089 * T + 0.000180108282532 * \
        T2 + 1.74532925199e-07 * T3
    # (h) Mean longitude of the sun
    h = 4.88162798259 + 628.331950894 * T + 5.23598775598e-06 * T2
    # (N) Longitude of the moon's ascending node in its orbit reckoned from
    # the referred equinox
    N = 4.52360161181 - 33.757146295 * T + 3.6264063347e-05 * T2 + \
        3.39369576777e-08 * T3
    cosN = np.cos(N)
    sinN = np.sin(N)

    # I (uppercase i) Inclination of the moon's orbit to the equator
    I = np.arccos(np.cos(ω) * np.cos(i) - np.sin(ω) * np.sin(i) *
                  cosN)
    # ν (nu) Longitude in the celestial equator of its intersection A with
    #  the moon's orbit
    ν = np.arcsin(np.sin(i) * sinN / np.sin(I))
    # t Hour angle of mean sun measured west-ward from the place of
    # observations
    t = np.radians(15. * (t0 - 12) - lon)

    # χ (chi) right ascension of meridian of place of observations reckoned
    # from A
    χ = t + h - ν
    # cos α (alpha) where α is defined in eq. 15 and 16
    cos_α = cosN * np.cos(ν) + sinN * np.sin(ν) * np.cos(ω)
    # sin α (alpha) where α is defined in eq. 15 and 16
    sin_α = sin(ω) * sinN / np.sin(I)
    # (α) α is defined in eq. 15 and 16
    α = 2 * np.arctan(sin_α / (1 + cos_α))
    # ξ (xi) Longitude in the moon's orbit of its ascending intersection
    # with the celestial equator
    ξ = N - α

    # σ (sigma) Mean longitude of moon in radians in its orbit reckoned
    # from A
    σ = s - ξ
    # l (lowercase el) Longitude of moon in its orbit reckoned from its
    # ascending intersection with the equator
    l = σ + 2 * e * np.sin(s - p) + (5. / 4) * e * e * np.sin(
        2 * (s - p)) + (15. / 4) * m * e * np.sin(s - 2 * h + p) + (
                11. / 8) * m * m * np.sin(2 * (s - h))

    # Solar Calculations #

    # p1 (p-one) Longitude of solar perigee
    # p1 = 281° 13' 15.0" + 6189.03" T + 1.63" T2 + 0.012" T3
    # Schureman [1941, pp. 162]
    p1 = 4.90822941839 + 0.0300025492114 * T + 7.85398163397e-06 * T2 \
        + 5.3329504922e-08 * T3
    # e1 (e-one) Eccentricity of the Earth's orbit
    e1 = 0.01675104 - 0.00004180 * T - 0.000000126 * T2
    # χ1 (chi-one) right ascension of meridian of place of observations
    # reckoned from the vernal equinox
    χ1 = t + h
    # l1 (lowercase-el(L) one) Longitude of sun in the ecliptic reckoned
    # from the vernal equinox
    l1 = h + 2 * e1 * np.sin(h - p1)
    # cosθ (theta) θ represents the zenith angle of the moon
    cosθ = sinλ * np.sin(I) * np.sin(l) + cosλ * (
            np.cos(0.5 * I) ** 2 * np.cos(l - χ) + np.sin(0.5 * I) ** 2 *
            np.cos(l + χ))
    # cosφ (phi) φ represents the zenith angle of the run
    cosφ = sinλ * sin(ω) * np.sin(l1) + cosλ * \
        (np.cos(0.5 * ω) ** 2 * np.cos(l1 - χ1) + np.sin(
         0.5 * ω) ** 2 * np.cos(l1 + χ1))

    # Distance Calculations #

    # (C) Distance parameter, equation 34
    # C**2 = 1/1( + 0.006738 sinλ ** 2)
    C = np.sqrt(1. / (1 + 0.006738 * sinλ ** 2))
    # (r) Distance from point P to the center of the Earth
    r = C * a + ht
    # a' (a prime) Distance parameter, equation 31
    aprime = 1. / (c * (1 - e * e))
    # a1' (a-one prime) Distance parameter, equation 31
    aprime1 = 1. / (c1 * (1 - e1 * e1))
    # (d) Distance between centers of the Earth and the moon
    d = 1. / ((1. / c) + aprime * e * np.cos(s - p) + aprime * e * e *
              np.cos(2 * (s - p)) + (15. / 8) * aprime * m * e * np.cos(
                s - 2 * h + p) + aprime * m * m * np.cos(2 * (s - h)))
    # (D) Distance between centers of the Earth and the sun
    D = 1. / ((1. / c1) + aprime1 * e1 * np.cos(h - p1))

    # (gm) Vertical component of tidal acceleration due to the moon
    # Equation (1):
    # gm = μMr/d^3 (3 * cos^2(θ) - 1) + 3/2 μMr/d^4 * (5 cos^3 θ - 3 cos(θ))
    gm = (μ * M * r / d ** 3) * (3 * cosθ ** 2 - 1) + (
         1.5 * (μ * M * r ** 2 / d ** 4) * (5 * cosθ ** 3 - 3 * cosθ))
    # (gs) Vertical component of tidal acceleration due to the sun
    gs = μ * S * r / D ** 3 * (3 * cosφ ** 2 - 1)

    print("Love factor: ", love)
    g0 = (gm + gs) * 1e3 * love

    # Returns Lunar, Solar, Total corrections in mGals
    return gm * 1e3 * love, gs * 1e3 * love, g0

