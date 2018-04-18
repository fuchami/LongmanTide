# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import pytest
from longmantide import solve_longman_tide


def test_array_calculation():
    lat = np.array([47.1234, 49.8901])
    lon = np.array([104.9903, 105.9901])
    alt = np.array([1609.3, 1700.1])
    time = np.array([datetime(2018, 4, 17, 12, 0, 0),
                     datetime(2018, 4, 17, 13, 0, 0)])

    lunar, solar, total = solve_longman_tide(lat, lon, alt, time)
    print("Lunar: {}\nSolar: {}\nTotal: {}".format(lunar, solar, total))


def test_static_location_tide():
    lat = np.array([40.7914])
    lon = np.array([282.1414])
    alt = np.array([370.])
    time = np.array([datetime(2015, 4, 23, 0, 0, 0)])

    gm, gs, g = solve_longman_tide(lat, lon, alt, time)
    print("gLunar: ", gm[0])
    print("gSolar: ", gs[0])
    print("gTotal: ", g[0])
    assert gm[0] == pytest.approx(0.0324029651226)
    assert gs[0] == pytest.approx(-0.0288682178454)
    assert g[0] == pytest.approx(0.00353474727722)






