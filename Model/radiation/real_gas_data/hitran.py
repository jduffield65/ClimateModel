import os
import numpy as np
# from ..constants import Avogadro, p_one_atmosphere, speed_of_light, h_planck, k_boltzmann, p_surface, p_toa
from Model.constants import Avogadro, p_one_atmosphere, speed_of_light, h_planck, k_boltzmann, p_surface, p_toa
from .specific_humidity import molecules
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter
from math import floor, ceil

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
data_folder = os.path.dirname(__file__) + '/HitranData/'
LookupTableFolder = data_folder + 'LookupTables/'
# could use gamma_self if on a planet with single molecule, but at the moment use gamma_air.
required_fields = ['molec_id', 'local_iso_id', 'nu', 'sw', 'elower', 'gamma_air', 'n_air']  # ,'gamma_self']

# reference values to compute gamma and sw given by hitran
p_reference = p_one_atmosphere
T_reference = 296


def load_molecule_data(molecule_name):
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
    return molecule_data


def get_consecutive_numbers(array):
    consec_array = []
    for k, g in groupby(enumerate(array), lambda i_x: i_x[0] - i_x[1]):
        consec_array.append(list(map(itemgetter(1), g)))
    return consec_array


def get_wavenumber_array(molecule_data, dwavenumber=10, bin_spacing=500, hist_thresh=100):
    """
    Produce histogram of number of lines in each wavenumber range of size bin_spacing, weighted by
    the strength of the lines. Cut off wavenumber is where weighted_histogram < hist_thresh.

    :param molecule_data:
    :param dwavenumber: cm^-1
    :param bin_spacing: cm^-1
    :param hist_thresh:
    :return:
    """
    weights = molecule_data['sw'].copy()
    # increase weights of small lines and decrease weights of large lines so count still important.
    weights[np.log10(weights) < -5] = 99
    weights[weights < 1] = 1
    weights[weights == 99] = 0.1
    weights[weights > 100] = 100

    bins = np.arange(molecule_data['nu'].min(), molecule_data['nu'].max() + bin_spacing, bin_spacing)
    hist_data = np.histogram(molecule_data['nu'], bins,  weights=weights)
    bins_below_thresh = np.where(hist_data[0] < hist_thresh)[0]
    consec_bins_below_thresh = get_consecutive_numbers(bins_below_thresh)

    # lower threshold is left hand side of first bin exceeding bin_thresh
    if not np.any(bins_below_thresh == 0):
        bin_min_index = 0
    else:
        bin_min_index = max(consec_bins_below_thresh[0])+1
    # upper threshold is right hand side of last bin exceeding bin_thresh
    if not np.any(bins_below_thresh == len(hist_data[0])-1):
        bin_max_index = len(hist_data[0])
    else:
        bin_max_index = min(consec_bins_below_thresh[-1])

    wavenumber_min = dwavenumber * round(floor(bins[bin_min_index] / dwavenumber))
    wavenumber_max = dwavenumber * round(ceil(bins[bin_max_index] / dwavenumber))
    wavenumber_array = np.arange(wavenumber_min, wavenumber_max+dwavenumber/2, dwavenumber)
    return wavenumber_array


def update_molecule_data(molecule_data, wavenumber_array):
    # only keep lines with wavenumber in wavenumber array
    keep_lines = np.where(np.logical_and(molecule_data['nu'] >= wavenumber_array.min(),
                                         molecule_data['nu'] <= wavenumber_array.max()))
    molecule_data_new = {key: molecule_data[key][keep_lines] for key in molecule_data}
    return molecule_data_new


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
    return gamma_reference * (p / p_reference) * (T_reference / T) ** n


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
    return s_reference * (T / T_reference) ** n * np.exp(-(h_planck * freq / k_boltzmann) * (1 / T - 1 / T_reference))


def lorentzian_profile(wave_number_array, wave_number_line_center, gamma):
    """

    :param wave_number_array: numpy array [n_p x n_wavenumber]
    :param wave_number_line_center: float
    :param gamma: [np x n_wavenumber]
    :return:
    """
    return (1 / np.pi) * gamma / (gamma ** 2 + (wave_number_array - wave_number_line_center) ** 2)


def wavenumbers_near_line(wavenumber_array, array_spacing, wavenumber_line_center, line_width, n_line_widths):
    """
    Finds indices of wavenumber_array that are near wavenumber_line_center (within n_line_widths).
    Required to compute the line shape more quickly i.e. dont use entire wavenumber range.
    i1 is min index of range. i2-1 is max index of range.

    :param wavenumber_array: numpy array [n_wavenumbers]
    :param array_spacing: float
    :param wavenumber_line_center: float
    :param line_width: float
    :param n_line_widths: integer
    :return:
    """
    centre_ind = np.abs(wavenumber_array - wavenumber_line_center).argmin()
    n_wavenumbers = int(n_line_widths * line_width / array_spacing)
    i1 = max(0, centre_ind - n_wavenumbers)
    i2 = min(np.size(wavenumber_array) - 1, centre_ind + n_wavenumbers) + 1
    return i1, i2


def single_line_absorption_coefficient(p, T, wavenumber_grid, line_data, d_wavenumber, n_line_widths):
    """

    :param p: numpy array [np]
    :param T: numpy array [np]
    :param wavenumber_grid: numpy array [np x n_wavenumber]
    :param line_data: dictionary
    :param d_wavenumber: float
    :param n_line_widths: integer
    :return:
    """
    gamma = gamma_extrapolate(p, T, line_data['gamma_air'], line_data['n_air'])
    line_strength = s_extrapolate(T, line_data['sw'], line_data['nu'], line_data['n_air'])
    i1, i2 = wavenumbers_near_line(wavenumber_grid[0, :], d_wavenumber, line_data['nu'], gamma.max(), n_line_widths)
    wavenumber_grid = wavenumber_grid[:, i1:i2]
    gamma_grid = np.repeat(gamma.reshape([-1, 1]), np.size(wavenumber_grid, 1), 1)
    line_shape = lorentzian_profile(wavenumber_grid, line_data['nu'], gamma_grid)
    return line_strength.reshape([-1, 1]) * line_shape, i1, i2


def get_absorption_coefficient(p, T, wavenumber_array, molecule_name, molecule_data=None, n_line_widths=1000):
    print('Loading in HITRAN data for ' + molecule_name)
    if molecule_data is None:
        molecule_data = load_molecule_data(molecule_name)

    wavenumber_grid = np.repeat(wavenumber_array.reshape([1, -1]), np.size(p), 0)
    absorption_coef_grid = np.zeros_like(wavenumber_grid)

    d_wavenumber = wavenumber_array[1] - wavenumber_array[0]
    # sum absorption coefficient contribution from each line
    for i in tqdm(range(len(molecule_data['nu']))):
        line_data = {key: molecule_data[key][i] for key in molecule_data}
        line_absorb_coef, i1, i2 = single_line_absorption_coefficient(p, T, wavenumber_grid, line_data,
                                                                      d_wavenumber, n_line_widths)
        absorption_coef_grid[:, i1:i2] += line_absorb_coef
    return absorption_coef_grid, molecule_data


def make_table(p_array, T_array, molecule_name, dwavenumber=10, n_line_widths=1000):
    output_file = LookupTableFolder + molecule_name + '.npy'
    if os.path.isfile(output_file) is True:
        raise ValueError('Lookuptable file already exists')
    molecule_data = load_molecule_data(molecule_name)
    wavenumber_array = get_wavenumber_array(molecule_data, dwavenumber)
    molecule_data = update_molecule_data(molecule_data, wavenumber_array)

    final_dict = {'p': p_array, 'T': T_array, 'nu': wavenumber_array}
    absorption_coef_grid = np.zeros((np.size(p_array), np.size(T_array), np.size(wavenumber_array)))
    for i in range(np.size(T_array)):
        print('Obtaining absorption coefficient ' + str(i + 1) + '/' + str(np.size(T_array)))
        T = np.ones_like(p_array) * T_array[i]
        absorption_coef_grid[:, i, :], _ = get_absorption_coefficient(p_array, T, wavenumber_array,
                                                                                  molecule_name, molecule_data,
                                                                                  n_line_widths)
    final_dict['absorption_coef'] = absorption_coef_grid
    np.save(data_folder + 'LookupTables/' + molecule_name + '.npy', final_dict)
    # read_dictionary = np.load(output_file, allow_pickle='TRUE').item()


def plot_absorption_coefficient(p_array, T_array, p_plot, wavenumber_array, absorption_grid, molecule_name):
    p_index = np.abs(p_array - p_plot).argmin()
    p_actual_plot = int(round(p_array[p_index]))
    T_actual_plot = int(round(T_array[p_index]))
    fig, ax = plt.subplots(1, 1)
    ax.plot(wavenumber_array, absorption_grid[p_index, :])
    ax.set_yscale('log')
    ax.set_ylim((10 ** -10, max(10 ** 6, absorption_grid.max())))
    ax.set_xlim(wavenumber_array.min(), wavenumber_array.max())
    ax.set_xlabel('Wavenumber cm$^{-1}$')
    ax.set_ylabel('Absorption coefficient (m$^2$/kg)')
    ax.set_title(molecule_name + ' at (' + str(T_actual_plot) + ' K, ' + str(p_actual_plot) + ' Pa), air-broadened')


if __name__ == '__main__':
    molecule_name = 'CO2'
    # wave_number_start = 25
    # wave_number_end = 3000
    dwave_number = 10
    # wavenumber_array = np.arange(wave_number_start, wave_number_end, dwave_number, dtype=float)
    p_array = np.logspace(np.log10(p_surface), np.log10(p_toa), 200)
    T_array = np.arange(250, 350 + 10, 20)
    make_table(p_array, T_array, molecule_name, dwave_number)
    # T_array = np.ones_like(p_array) * 260
    # absorb_coef_grid, _ = get_absorption_coefficient(p_array, T_array, wavenumber_array, molecule_name)
    # plot_absorption_coefficient(p_array, T_array, p_surface, wavenumber_array, absorb_coef_grid, molecule_name)
    # plt.show()