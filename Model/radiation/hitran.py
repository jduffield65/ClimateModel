import os
import numpy as np
from ..constants import Avogadro, p_one_atmosphere, speed_of_light, h_planck, k_boltzmann
"""Format of molecule_file data is explained in OutputExplained.txt:
we are interested in following columns:
molec_id: integer HITRAN id of molecule
local_iso_id: Integer ID of a particular Isotopologue (1 = most abundant)
nu: Transition wavenumber (cm^-1)
sw: Line intensity at 296K (cm^-1/(molec.cm^-2))
elower: Lower-state energy (cm^-1)
gamma_air: Air-broadened Lorentzian half-width at half-maximum at p = 1 atm and T = 296 K (cm^-1.atm^-1)
gamma_self: Self-broadened HWHM at 1 atm pressure and 296 K (cm^-1.atm^-1)
n_air: Temperature exponent for the air-broadened HWHM
"""
# os bit gets folder two above
data_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/HitranData/'
# could use gamma_self if on a planet with single molecule, but at the moment use gamma_air.
required_fields = ['molec_id', 'local_iso_id', 'nu', 'sw', 'elower', 'gamma_air', 'n_air'] # ,'gamma_self']

# list hitran id and molecular mass in gmol^-1 for some molecules
molecules = {}
molecules['H20'] = {'hitran_id': 1, 'M': 18}
molecules['CO2'] = {'hitran_id': 2, 'M': 44}
molecules['O3'] = {'hitran_id': 3, 'M': 48}
molecules['CH4'] = {'hitran_id': 6, 'M': 16}

# reference values to compute gamma and sw given by hitran
p_reference = p_one_atmosphere
T_reference = 296

def s_conversion(s, M):
    """
    converts line intensity from units of (cm^-1/(molec.cm^-2)) to units of (cm^-1.(m^2.kg^-1))
    :param s: float
        line intensity in (cm^-1/(molec.cm^-2)) units
    :param M: float
        Atomic mass of gass in units of gmol^-1
    """
    return 0.1 * Avogadro / M * s


def gamma_extrapolate(p, T, gamma_reference, n):
    """
    equation 4.61 in Principles of Planetary Climate
    full version given by equation 6 here: https://hitran.org/docs/definitions-and-units/
    Probably more accurate. HAVE NOT USED PARTIAL PRESSURES
    Given a reference line width, find it at a given temperature and pressure.

    :param p: numpy array [n_p]
    :param T: numpy array [n_p]
    :param gamma_reference: float
    :param n: float
    """
    return gamma_reference * (p / p_reference) * (T_reference / T)**n


def s_extrapolate(T, s_reference, wave_number_line_center, n):
    """
    equation 4.62 in Principles of Planetary Climate
    full version given by equation 4 here: https://hitran.org/docs/definitions-and-units/
    Probably more accurate. HAVE NOT USED PARTITION FUNCTION
    Given a reference line intensity, find it at a given temperature and pressure.

    :param T: numpy array [np]
    :param s_reference: float
    :param wave_number_line_center: float (cm^-1)
    :param n: float
    """
    # convert wavenumber in cm^-1 to frequency in s^-1
    freq = 100 * wave_number_line_center * speed_of_light
    return s_reference * (T / T_reference)**n * np.exp(-(h_planck * freq / k_boltzmann) * (1/T - 1/T_reference))


def lorentzian_profile(wave_number_array, wave_number_line_center, gamma):
    """

    :param wave_number_array: numpy array [n_p x n_wavenumber]
    :param wave_number_line_center: float
    :param gamma: [np x n_wavenumber]
    :return:
    """
    return (1/np.pi) * gamma / (gamma**2 + (wave_number_array - wave_number_line_center)**2)


def single_line_absorption_coefficient(p, T, wavenumber_array, line_data):
    gamma = gamma_extrapolate(p, T, line_data['gamma_air'], line_data['n_air'])
    line_strength = s_extrapolate(T, line_data['sw'], line_data['nu'], line_data['n_air'])
    gamma_grid, wavenumber_grid = np.meshgrid(gamma, wavenumber_array)  #NEXT THING TO PROGRAMME!!
    line_shape = lorentzian_profile(wavenumber_grid, line_data['nu'], gamma_grid)
    return line_strength * line_shape


def get_absorption_coefficient(p, T, wavenumber_array, molecule_name, molecule_data=None):
    if molecule_data is not None:
        molecule_file = data_folder + molecule_name + '.txt'
        molecule_data = np.genfromtxt(molecule_file, names=True)
        # ensure only correct molecule and most abundant molecule are loaded
        correct_isotope = np.where(np.logical_and(molecule_data['molec_id'] == molecules[molecule_name]['hitran_id'],
                                                  molecule_data['local_iso_id'] == 1))
        molecule_data = molecule_data[correct_isotope]
        # only keep required data
        molecule_data = {field: molecule_data[field] for field in required_fields[2:]}
        # convert units of line strength
        molecule_data['sw'] = s_conversion(molecule_data['sw'], molecules[molecule_name]['M'])
    abs_coef = np.zeros(np.size(p), np.size(wavenumber_array))

    # sum absorption coefficient contribution from each line
    for i in range(len(molecule_data['nu'])):
        line_data = {key: molecule_data[key][i] for key in molecule_data}
        abs_coef += single_line_absorption_coefficient(p, T, wavenumber_array, line_data)
    return abs_coef


molecule_name = 'CO2'
molecule_file = data_folder + molecule_name + '.txt'
molecule_data = np.genfromtxt(molecule_file, names=True)
# ensure only correct molecule and most abundant molecule are loaded
correct_isotope = np.where(np.logical_and(molecule_data['molec_id'] == hitran_id[molecule_name],
                                          molecule_data['local_iso_id'] == 1))
molecule_data = molecule_data[correct_isotope]
# only keep required data
molecule_data = {field: molecule_data[field] for field in required_fields[2:]}
