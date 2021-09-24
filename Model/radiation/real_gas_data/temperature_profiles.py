"""
Each function returns the Temperature (Kelvin)
at each pressure level.

PAPERS THAT DATA WAS FOUND FROM:
earth_temp:
Whole Atmosphere Climate Change: Dependence
on Solar Activity (Stanley C. Solomon)
Figure 3a is used for the atmosphere temperature profile earth_temp
"""
from .specific_humidity import p_altitude_convert
import numpy as np
from scipy.interpolate import interp1d


def earth_temp(p):
    """
    Gets temperatures of earth at pressure levels p.
    :param p: numpy array
    """
    h_values = np.array([0, 12, 19, 21, 30, 40, 46, 50, 70, 79, 81, 88, 99, 140]) * 1000
    T_values = np.array([288, 210, 205, 215, 226, 250, 260, 260, 210, 199, 199, 202, 195, 610])
    h = p_altitude_convert(p=p)
    interp_func = interp1d(h_values, T_values)
    T = np.zeros_like(p)
    T[h <= h_values[-1]] = interp_func(h[h <= h_values[-1]])
    T[h > h_values[-1]] = T_values[-1]
    return T


def fixed_tropopause_temp(p, h_tropopause=19, T_tropopause=205, T_ground=288):
    """
    This has a troposphere but above this, the temperature is fixed at the tropopause temperature.
    :param p: numpy array
    :param h_tropopause: height of tropopause (km)
    :param T_tropopause: temperature of tropopause (K)
    :param T_ground: temperature of ground (K)
    """
    h_values = np.array([0, h_tropopause, 140]) * 1000
    T_values = np.array([288, T_tropopause, T_tropopause])
    h = p_altitude_convert(p=p)
    interp_func = interp1d(h_values, T_values)
    T = np.zeros_like(p)
    T[h <= h_values[-1]] = interp_func(h[h <= h_values[-1]])
    T[h > h_values[-1]] = T_values[-1]
    return T
