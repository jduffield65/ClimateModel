from ..constants import g, c_p_dry, sigma, p_surface, p_toa, F_sun, R_specific
import numpy as np
from itertools import groupby
from operator import itemgetter


def convective_adjustment(p, T, lapserate=g/c_p_dry):
    """
    Just runs function convective_adjustment_single for each latitude.

    :param p: numpy array [nz]
        p[0] = p[surface] > p[-1] = p[top of atmosphere]
    :param T: numpy array [nz x ny]
        T[i, j] is temperature at pressure level p[i] at latitude index j
    :param lapserate: desired lapse rate, optional.
        lapserate = -dT/dz in units of K/m
        default: g/c_p_dry i.e. dry adiabat
    """
    if len(np.shape(p)) > 1:
        raise ValueError('Pressure has wrong dimension, must be only in the z dimension')
    if p[1] > p[0]:
        # ensure pressure is descending
        invert_p = True
        p = p[::-1]
        T = T[::-1, :]
    else:
        invert_p = False
    for i in range(np.shape(T)[1]):
        T[:, i] = convective_adjustment_single(p, T[:, i], lapserate)
    if invert_p:
        # revert temperature to previous order.
        T = T[::-1, :]
    return T


def convective_adjustment_single(p, T, lapserate=g/c_p_dry):
    """
    Changes temperature profile wherever dT/dz < -lapserate so
    convective stability dT/dz >= -lapserate is reached everywhere.

    :param p: numpy array [nz]
        p[0] = p[surface] > p[-1] = p[top of atmosphere]
    :param T: numpy array [nz]
        T[i] is temperature at pressure level p[i]
    :param lapserate: desired lapse rate, optional.
        lapserate = -dT/dz in units of K/m
        default: g/c_p_dry i.e. dry adiabat
    :return:
        new Temperature profile
    """
    p_reference = p_surface
    alpha = R_specific * lapserate / g
    theta = get_theta(T, p, p_reference, alpha)
    theta_diff = np.ediff1d(theta)
    theta_diff = np.concatenate((theta_diff, [theta_diff[-1]]))
    small = 1e-10  # so don't repeat with tiny increase
    unstable_levels = np.where(theta_diff < -small)[0] # negative as pressure is decreasing.
    while len(unstable_levels) > 0:
        enthalpy = get_enthalpy(T, p)
        unstable_groups = []
        for k, j in groupby(enumerate(unstable_levels), lambda x: x[0] - x[1]):
            group = (map(itemgetter(1), j))
            group = list(map(int, group))
            unstable_groups.append(group)

        for unstable_group in unstable_groups:
            adjust_theta = {'lower': {}, 'upper': {}}
            # lower pressure level of unstable region
            min_pressure_to_change = unstable_group[-1]+1
            adjust_theta['lower']['theta'] = theta[min_pressure_to_change]
            low_theta_levels = np.where(theta < adjust_theta['lower']['theta'])[0]
            low_theta_levels = low_theta_levels[low_theta_levels < min_pressure_to_change]
            if len(low_theta_levels) == 0:
                max_pressure_to_change = 0
            else:
                max_pressure_to_change = nearest_value_in_array(low_theta_levels, min_pressure_to_change) + 1
            adjust_theta['lower']['change_levels'] = np.arange(max_pressure_to_change, min_pressure_to_change+1)
            # upper pressure level of unstable region
            max_pressure_to_change = unstable_group[0]
            adjust_theta['upper']['theta'] = theta[max_pressure_to_change]
            high_theta_levels = np.where(theta > adjust_theta['upper']['theta'])[0]
            high_theta_levels = high_theta_levels[high_theta_levels > max_pressure_to_change]
            if len(high_theta_levels) == 0:
                min_pressure_to_change = len(p) - 1
            else:
                min_pressure_to_change = nearest_value_in_array(high_theta_levels, max_pressure_to_change)
            adjust_theta['upper']['change_levels'] = np.arange(max_pressure_to_change, min_pressure_to_change + 1)

            for key in adjust_theta:
                theta_adjust = theta.copy()
                theta_adjust[adjust_theta[key]['change_levels']] = adjust_theta[key]['theta']
                adjust_theta[key]['T'] = get_T(theta_adjust, p, p_reference, alpha)
                adjust_theta[key]['enthalpy'] = get_enthalpy(adjust_theta[key]['T'], p)

            # combine upper and lower profiles as to conserve energy
            beta = (enthalpy - adjust_theta['lower']['enthalpy']) / (
                    adjust_theta['upper']['enthalpy'] - adjust_theta['lower']['enthalpy'])
            T = beta * adjust_theta['upper']['T'] + (1-beta) * adjust_theta['lower']['T']
            # update theta for next iteration.
            theta = get_theta(T, p, p_reference, alpha)
        theta_diff = np.ediff1d(theta)
        theta_diff = np.concatenate((theta_diff, [theta_diff[-1]]))
        unstable_levels = np.where(theta_diff < -small)[0]  # negative as pressure is decreasing.

    return T


def nearest_value_in_array(array, value):
    return array[np.abs(array - value).argmin()]


def get_theta(T, p, p_reference, alpha):
    return T / (p/p_reference)**alpha


def get_T(theta, p, p_reference, alpha):
    return theta * (p/p_reference)**alpha


def get_enthalpy(T, p):
    # enthalpy is proportional to integral of temperature with respect to pressure
    return -np.trapz(T, p)
