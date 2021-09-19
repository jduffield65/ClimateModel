"""
Each function returns the specific humididty, q = rho_molecule / rho_air,
at each pressure level using typical Earth values.
"""
import numpy as np

M_air = 28.97  # molar mass of air in gmol^-1


def humidity_from_ppmv(conc_ppmv, molecule_name):
    """
    Given molar concentration in parts per million by volume,
    specific humidity is returned

    :param conc_ppmv:
    :param molecule_name:
    :return:
    """
    return conc_ppmv / 10 ** 6 * molecules[molecule_name]['M'] / M_air


def co2(p):
    """
    Using figure 9b from paper:
    Carbon dioxide trends in the mesosphere and lower thermosphere
    by Liying Qian
    q is constant at p > p_thresh and falls off linearly with log10(p) below p_thresh.
    
    :param p: numpy array
    :return: q
    """
    q_ppmv_surface = 370
    q = np.ones_like(p) * q_ppmv_surface
    q_ppmv_toa = 150
    p_thresh = 0.5
    p_toa = 0.01
    gradient = (q_ppmv_surface - q_ppmv_toa) / (np.log10(p_thresh) - np.log10(p_toa))
    intercept = q_ppmv_surface - gradient * np.log10(p_thresh)
    q[p < p_thresh] = intercept + gradient * np.log10(p[p < p_thresh])
    q[q < 0] = 0
    q = humidity_from_ppmv(q, 'CO2')
    return q

# list hitran id and molecular mass in gmol^-1 for some molecules as well as humidity functions
molecules = {}
molecules['H20'] = {'hitran_id': 1, 'M': 18}
molecules['CO2'] = {'hitran_id': 2, 'M': 44, 'humidity': co2}
molecules['O3'] = {'hitran_id': 3, 'M': 48}
molecules['CH4'] = {'hitran_id': 6, 'M': 16}
