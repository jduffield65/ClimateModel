"""
Each function returns the specific humididty, q = rho_molecule / rho_air,
at each pressure level using typical Earth values.

PAPER THAT DATA WAS FOUND FROM:
Whole Atmosphere Climate Change: Dependence
on Solar Activity (Stanley C. Solomon)
Figure 1 contains altitude profiles of all 4 gases.
We use approximations of red curve (2003) here.
Figure 4 used to convert between altitude and pressure.
"""
import numpy as np
from ...constants import p_surface_earth
from scipy.interpolate import interp1d

M_air = 28.97  # molar mass of air in gmol^-1


def p_altitude_convert(altitude=None, p=None):
    """
    if give altitude, will return pressure (Pa)
    if give pressure, will return altitude (m)
    :param altitude: numpy array (meters)
    :param p: numpy array (Pa)
    """
    # below 90km, assume linear correspondence
    h1 = 0.0
    p1_log = np.log10(p_surface_earth)
    h2 = 90000.0
    p2_log = -1.0
    gradient = (p2_log - p1_log) / (h2 - h1)
    # above 90km, assume linear correspondence
    h3 = 130000.0
    p3_log = -3.0
    gradient2 = (p3_log - p2_log) / (h3 - h2)
    if p is None:
        log_p = np.zeros_like(altitude, dtype=float)
        log_p[altitude <= h2] = p1_log + gradient * altitude[altitude <= h2]
        log_p[altitude > h2] = p2_log + gradient2 * (altitude[altitude > h2] - h2)
        return 10 ** log_p
    elif altitude is None:
        log_p = np.log10(p)
        altitude = np.zeros_like(p, dtype=float)
        altitude[log_p >= p2_log] = (log_p[log_p >= p2_log] - p1_log) / gradient
        altitude[log_p < p2_log] = (log_p[log_p < p2_log] - p2_log) / gradient2 + h2
        return altitude


def humidity_from_ppmv(conc_ppmv, molecule_name):
    """
    Given molar concentration in parts per million by volume,
    specific humidity is returned (kg / kg)

    :param conc_ppmv: numpy array
    :param molecule_name: string e.g. 'CO2'
    """
    return conc_ppmv / 10 ** 6 * molecules[molecule_name]['M'] / M_air


def ppmv_from_humidity(humidity, molecule_name):
    """
    Given specific humidity (kg / kg),
     molar concentration in parts per million by volume is returned

    :param humidity: numpy array
    :param molecule_name: string e.g. 'CO2'
    """
    return humidity * 10 ** 6 * M_air / molecules[molecule_name]['M']


def co2(p, q_surface=370, h_change=80000):
    """
    q is constant for h < h_change and falls off linearly with h above h_change
    
    :param p: numpy array
    :param q_surface: surface humidity (ppmv)
    :param h_change: float, optional (meters)
        altitude above which q starts decreasing.
        default: 80000
    :return: q
    """
    h = p_altitude_convert(p=p)
    if q_surface == 0:
        q = np.zeros_like(p)
    else:
        q = np.ones_like(p) * q_surface
        h_toa = 120000
        q_toa = 60
        gradient = (q_surface - q_toa) / (h_change - h_toa)
        intercept = q_surface - gradient * h_change
        q[h > h_change] = intercept + gradient * h[h > h_change]
        q[q < 0] = 0
        q = humidity_from_ppmv(q, 'CO2')
    return q


def ch4(p, scale_factor=1):
    """
    interpolate q from plot in paper
    :param p: numpy array
    :param scale_factor: float, optional.
        Surface q gets multiplied by this amount, q further from surface gets multiplied by lesser amount
        q at top of atmosphere always remains the same.
        default: 1
    """
    """Get interpolation function from data"""
    h_values = np.array([0, 10, 17, 22, 28, 50, 68, 80, 90]) * 1000
    q_values = np.array([1.75, 1.75, 1.68, 1.32, 1.19, 0.4, 0.19, 0.04, 0])
    # mod_factor = abs(scale_factor - 1) * (h_values[-1] - h_values) / h_values[-1]
    # mod_factor = 1 + np.sign(scale_factor - 1) * mod_factor
    mod_factor = scale_factor
    q_values = q_values * mod_factor
    if scale_factor == 0:
        q = np.zeros_like(p)
    else:
        q_values[1] = q_values[0]  # to maintain constant value up to certain h
        q_values[q_values > q_values[0]] = q_values[0]  # to keep maxima at surface
        interp_func = interp1d(h_values, q_values)
        """interpolate given pressure values"""
        h = p_altitude_convert(p=p)
        q = np.zeros_like(p)
        q[h < h_values.max()] = interp_func(h[h < h_values.max()])
        q[q < 0] = 0
        q = humidity_from_ppmv(q, 'CH4')
    return q


def h2o(p, scale_factor=1):
    """
    interpolate log(q) from plot in paper
    :param p: numpy array
    :param scale_factor: float, optional.
        Surface q gets multiplied by this amount, q further from surface gets multiplied by lesser amount
        q at top of atmosphere always remains the same.
        default: 1
    """
    """Get interpolation function from data"""
    # ppmv values from plot in paper
    h_values = np.arange(0, 90, 5) * 1000
    q_values = np.array([20000, 2500, 250, 12, 4, 4.3, 4.9, 5.1, 5.7, 5.9, 6, 6.1, 6, 5.8, 5, 4, 2.5, 1])
    # mod_factor = abs(scale_factor - 1) * (h_values[-1] - h_values) / h_values[-1]
    # mod_factor = 1 + np.sign(scale_factor - 1) * mod_factor
    mod_factor = scale_factor
    if scale_factor == 0:
        q = np.zeros_like(p)
    else:
        q_values = q_values * mod_factor
        q_values[q_values > q_values[0]] = q_values[0]  # to keep maxima at surface
        interp_func = interp1d(h_values, np.log10(q_values))
        """interpolate given pressure values"""
        h = p_altitude_convert(p=p)
        q = np.zeros_like(p)
        q[h < h_values.max()] = 10 ** interp_func(h[h < h_values.max()])
        q[q < 0] = 0
        q = humidity_from_ppmv(q, 'H2O')
    return q


def o3(p, scale_factor=1):
    """
    interpolate q from plot in paper
    :param p: numpy array
    :param scale_factor: float, optional.
        All q values away from top of atmosphere and surface get multiplied by this amount
        default: 1
    """
    """Get interpolation function from data"""
    h_values = np.sort(np.concatenate((np.arange(0, 125, 5), np.array([32, 78, 92])))) * 1000
    q_values = np.array([0.05, 0.07, 0.09, 0.25, 1.8, 5.25, 7.8, 7.9, 7.85, 6, 3.8, 2.4, 1.6, 1, 0.75, 0.3, 0.15, 0.1,
                         0.15, 0.8, 1.75, 1.8, 1.7, 1, 0.3, 0.07, 0.05, 0])
    if scale_factor == 0:
        q = np.zeros_like(p)
    else:
        # q_values[3:-3] = q_values[3:-3] * scale_factor
        q_values = q_values * scale_factor
        interp_func = interp1d(h_values, q_values)
        """interpolate given pressure values"""
        h = p_altitude_convert(p=p)
        q = np.zeros_like(p)
        q[h < h_values.max()] = interp_func(h[h < h_values.max()])
        q[q < 0] = 0
        q = humidity_from_ppmv(q, 'O3')
    return q


# list hitran id and molecular mass in gmol^-1 for some molecules as well as humidity functions
molecules = {'H2O': {'hitran_id': 1, 'M': 18, 'q': h2o, 'q_args': (1,)},
             'CO2': {'hitran_id': 2, 'M': 44, 'q': co2, 'q_args': (370, 80000)},
             'O3': {'hitran_id': 3, 'M': 48, 'q': o3, 'q_args': (1,)},
             'CH4': {'hitran_id': 6, 'M': 16, 'q': ch4, 'q_args': (1,)}}
