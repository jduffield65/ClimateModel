import os
import numpy as np
from Model.constants import Avogadro, p_one_atmosphere, speed_of_light, h_planck, k_boltzmann,\
    p_surface_earth, p_toa_earth
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
data_folder = os.path.dirname(__file__) + '/HitranData/'  # raw data here
LookupTableFolder = data_folder + 'LookupTables/'  # where data saved to
# could use gamma_self if on a planet with single molecule, but at the moment use gamma_air.
required_fields = ['molec_id', 'local_iso_id', 'nu', 'sw', 'elower', 'gamma_air', 'n_air']  # ,'gamma_self']

# reference values to compute gamma and sw given by hitran
p_reference = p_one_atmosphere
T_reference = 296
# Data saved to LookupTableFolder is [np x nT x n_nu] grid of absorption coefficients
# pressure values used are table_p_values by default. Not a problem if different between different molecules
# Temperature values used are table_T_values. Not a problem if different between different molecules
# wavenumber values have a separation of dnu. Important to keep dnu the same for all molecules.
table_p_values = np.logspace(np.log10(p_surface_earth), np.log10(p_toa_earth), 200)  # up to 20Pa
table_T_values = np.arange(250, 350 + 10, 20)
table_dnu = 10


def load_molecule_data(molecule_name):
    """
    loads molecule data from hitran file, returns required data in the form of dictionary
    where keys are those in required_fields list.

    :param molecule_name: string e.g. 'CO2'
    :return:
    """
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
    """
    finds clusters of consecutive values in an array
    """
    consec_array = []
    for k, g in groupby(enumerate(array), lambda i_x: i_x[0] - i_x[1]):
        consec_array.append(list(map(itemgetter(1), g)))
    return consec_array


def get_wavenumber_array(molecule_data, dwavenumber=10, bin_spacing=500, hist_thresh=100, n_line_widths=1000):
    """
    Produce histogram of number of lines in each wavenumber range of size bin_spacing, weighted by
    the strength of the lines. Cut off wavenumber is where weighted_histogram < hist_thresh.

    :param molecule_data: dictionary
    :param dwavenumber: float (cm^-1)
    :param bin_spacing: float (cm^-1)
    :param hist_thresh: float
    :param n_line_widths: integer.
        number of line widths to keep of lines on extremities of wavenumber range.
    """
    weights = molecule_data['sw'].copy()
    # increase weights of small lines and decrease weights of large lines so count still important.
    weights[np.log10(weights) < -5] = 99
    weights[weights < 1] = 1
    weights[weights == 99] = 0.1
    weights[weights > 100] = 100

    bins = np.arange(molecule_data['nu'].min()-n_line_widths*molecule_data['gamma_air'][molecule_data['nu'].argmin()],
                     molecule_data['nu'].max()+n_line_widths*molecule_data['gamma_air'][molecule_data['nu'].argmax()]
                     + bin_spacing-2, bin_spacing)
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
    """
    reduce size of molecule_data by only considering lines in given wavenumber range
    :param molecule_data: dictionary
    :param wavenumber_array: numpy array
    """
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
    8 here: https://hitran.org/docs/definitions-and-units/
    ignore pressure shift correction of line central wavenumber
    :param wave_number_array: numpy array [n_p x n_nu]
    :param wave_number_line_center: float
    :param gamma: [np x n_nu]
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
    Returns absorption coefficient for single line aswell as indices (i1:i2) of wavenumber_grid
    where line covers.
    :param p: numpy array [np]
    :param T: numpy array [np]
    :param wavenumber_grid: numpy array [np x n_wavenumber]
    :param line_data: dictionary
    :param d_wavenumber: float
    :param n_line_widths: integer, number of line widths to keep for each spectral line
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
    """
    Gets absorption coefficient grid [np x n_nu] summing up all lines of a molecule
    :param p: numpy array [np]
    :param T: numpy array [np]
    :param wavenumber_array: [n_nu]
    :param molecule_name: string e.g. 'CO2'
    :param molecule_data: if None, loads in data from file.
    :param n_line_widths: integer, number of line widths to keep for each spectral line
    """
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
    return absorption_coef_grid


def ozone_UV(wavenumber_array, p_array, T_array):
    """
    This reads in separate data file which has absorption coefficient in some of UV range at 273 K
    Then initialises absorption grid so that it includes this data.
    Assumes in this range, absorption coefficient is independent of temperature and pressure.

    :param wavenumber_array: numpy array [n_nu_IR]
        nu values in IR regime
    :param p_array: numpy array [np]
    :param T_array: numpy array [nT]
    :return:
        wavenumber_array [n_nu] and absorption grid [np x nT x n_nu]
    """
    """Get UV data"""
    # read in UV absorption cross section data
    file = data_folder + 'O3_UV_273.xsc'
    header = open(file).readline().rstrip() # [molecule, nu_min, nu_max,
    header = list(header.split('\t'))  # ensure first line is tab separated first
    min_nu = float(header[1])  # minimum wavenumber (cm-1)
    max_nu = float(header[2])  # maximum wavenumber (cm-1)
    N_nu = int(header[3])  # number of datapoints
    T = float(header[4])  # Temperature
    nu = np.linspace(min_nu, max_nu, N_nu)
    d_nu_raw_data = nu[1] - nu[0]
    absorption_coef = np.genfromtxt(file, skip_header=1).flatten()[:-1]  # in cm^2/molecule (last value is 0)
    absorption_coef = s_conversion(absorption_coef, molecules['O3']['M'])  # in m^2/kg

    # extrapolate, assuming symmetric about maxima
    max_ind = absorption_coef.argmax()
    repeat_end_ind = np.where(absorption_coef < absorption_coef[-1])[0]  # where absorption is less than final value
    repeat_end_ind = repeat_end_ind[repeat_end_ind < max_ind][-1]  # also less than max index
    repeat_nu = nu[:repeat_end_ind+1] - nu.min() + d_nu_raw_data + nu[-1]
    repeat_absorption = absorption_coef[:repeat_end_ind+1][::-1]
    nu = np.concatenate((nu, repeat_nu))
    absorption_coef = np.concatenate((absorption_coef, repeat_absorption))

    # get average at every 10cm^-1
    d_nu_target = int(round((wavenumber_array[1] - wavenumber_array[0])))  # ensure odd
    nu_convolve = np.convolve(nu, np.ones(d_nu_target+1)/(d_nu_target+1), mode='valid')
    absorption_coef_convolve = np.convolve(absorption_coef, np.ones((d_nu_target+1))/(d_nu_target+1), mode='valid')
    use = divmod(nu_convolve, d_nu_target)[1] == 0
    nu_final = nu_convolve[use]
    absorption_coef_final = absorption_coef_convolve[use]
    absorption_coef_final[0] = 10**-15  # very small so wavenumbers between UV and IR assigned value of 0.

    """append UV data to IR data"""
    if nu_final[0] < wavenumber_array[-1]:
        raise ValueError('UV and IR wavenumber regions overlap')
    wavenumber_final = np.concatenate((wavenumber_array, nu_final))
    absorption_coef_grid = np.zeros((np.size(p_array), np.size(T_array), np.size(wavenumber_final)))
    uv_nu_index = np.where(wavenumber_final.reshape(-1, 1) == nu_final)[0]
    # have same UV absorption for all pressures and temperatures
    absorption_coef_grid[:, :, uv_nu_index] = absorption_coef_final
    return wavenumber_final, absorption_coef_grid


def make_table(molecule_name, p_array=table_p_values, T_array=table_T_values,
               dwavenumber=table_dnu, n_line_widths=1000, wavenumber_array=None):
    """
    Makes [np x nT x n_nu] absorption coefficient lookup table and saves it to LookupTableFolder

    :param molecule_name: string e.g. 'CO2'
    :param p_array: numpy array [np]
    :param T_array: numpy array [nT]
    :param dwavenumber: float
    :param n_line_widths: integer, number of line widths to keep for each spectral line
    :param wavenumber_array: numpy array [n_nu] or None.
        If None, will work out range required to keep main lines.
    """
    if isinstance(molecule_name, dict):
        molecule_data = molecule_name
        molecule_name = 'custom'
    output_file = LookupTableFolder + molecule_name + '.npy'
    if os.path.isfile(output_file) is True:
        raise ValueError('Lookuptable file already exists')
    if molecule_name != 'custom':
        molecule_data = load_molecule_data(molecule_name)
    if wavenumber_array is None:
        wavenumber_array = get_wavenumber_array(molecule_data, dwavenumber, n_line_widths=n_line_widths)
    molecule_data = update_molecule_data(molecule_data, wavenumber_array)
    if molecule_name == 'O3':
        # add uv data for ozone
        wavenumber_array, absorption_coef_grid = ozone_UV(wavenumber_array, p_array, T_array)
    else:
        absorption_coef_grid = np.zeros((np.size(p_array), np.size(T_array), np.size(wavenumber_array)))
    final_dict = {'p': p_array, 'T': T_array, 'nu': wavenumber_array}
    for i in range(np.size(T_array)):
        print('Obtaining absorption coefficient ' + str(i + 1) + '/' + str(np.size(T_array)))
        T = np.ones_like(p_array) * T_array[i]
        absorption_coef_grid[:, i, :] += get_absorption_coefficient(p_array, T, wavenumber_array,
                                                                    molecule_name, molecule_data, n_line_widths)
    final_dict['absorption_coef'] = absorption_coef_grid
    np.save(data_folder + 'LookupTables/' + molecule_name + '.npy', final_dict)
    # read_dictionary = np.load(output_file, allow_pickle='TRUE').item()


def plot_absorption_coefficient(molecule_name, p_plot, T_plot, ax=None):
    """
    Plot absorption coefficient vs wavenumber for a specific pressure and temperature

    :param molecule_name: string e.g. 'CO2'
    :param p_plot: float
    :param T_plot: float
    """
    output_file = LookupTableFolder + molecule_name + '.npy'
    dict = np.load(output_file, allow_pickle='TRUE').item()
    p_index = np.abs(dict['p'] - p_plot).argmin()
    T_index = np.abs(dict['T'] - T_plot).argmin()
    absorption_coef = dict['absorption_coef'][p_index, T_index]
    p_actual_plot = int(round(dict['p'][p_index]))
    T_actual_plot = int(round(dict['T'][T_index]))
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot(dict['nu'], absorption_coef)
    ax.set_yscale('log')
    ax.set_ylim((10 ** -10, max(10 ** 6, absorption_coef.max())))
    ax.set_xlim(dict['nu'].min(), dict['nu'][np.where(absorption_coef > 10**-10)[0][-1]])
    ax.set_xlabel('Wavenumber cm$^{-1}$')
    ax.set_ylabel('Absorption coefficient (m$^2$/kg)')
    ax.set_title(molecule_name + ' at (' + str(T_actual_plot) + ' K, ' + str(p_actual_plot) + ' Pa), air-broadened')
