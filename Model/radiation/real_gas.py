from .real_gas_data.hitran import LookupTableFolder
from ..constants import h_planck, speed_of_light, k_boltzmann, g, T_sun, R_sun, AU, p_surface, p_toa, sigma
from .real_gas_data.specific_humidity import molecules
from .grey import GreyGas
import numpy as np
from math import ceil, floor
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d


def B_freq(freq, T):
    """
    Planck function in terms of frequency (s^-1)
    sigmaT^4 = integral(pi * B_freq) dfreq

    :param freq:
        unit = s^-1
    :param T:
    :return:
    """
    #Large = 1000
    #u = min(h_planck * freq / (k_boltzmann * T), Large) # To prevent overflow
    u = h_planck * freq / (k_boltzmann * T)
    return (2. * h_planck * freq ** 3 / speed_of_light ** 2) / (np.exp(u) - 1.)


def B_wavenumber(nu, T):
    """
    Planck function in terms of wavenumber (cm^-1)
    sigmaT^4 = integral(pi * B_freq) dnu

    :param nu:
        unit = cm^-1
    :param T:
    :return:
    """
    dfreq_dwavenumber = 100 * speed_of_light
    freq = 100 * nu * speed_of_light
    return dfreq_dwavenumber * B_freq(freq, T)


def get_isothermal_temp(albedo, F_stellar=None, T_star=None, R_star=None, star_planet_dist=None):
    if F_stellar is None:
        F_stellar = sigma * T_star**4 * R_star**2 / star_planet_dist **2
    return np.power(F_stellar / sigma * (1-albedo) / 4, 1 / 4)


def get_absorption_coef(p, T, nu, absorb_coef_dict):
    """

    :param p: [np]
    :param T: [np]
    :param nu: [n_wavenumber]
    :param absorb_coef_dict: dictionary
    :return:
        absorb_coef [np x n_wavenumber]
    """
    p_dict_ind = np.abs(p.reshape(-1, 1) - absorb_coef_dict['p'].reshape(-1, 1).transpose()).argmin(axis=1)
    T_dict_ind = np.abs(T.reshape(-1, 1) - absorb_coef_dict['T'].reshape(-1, 1).transpose()).argmin(axis=1)
    nu_dict_ind = np.abs(nu.reshape(-1, 1) -
                                 absorb_coef_dict['nu'].reshape(-1, 1).transpose()).argmin(axis=1)
    absorb_coef_all_nu = absorb_coef_dict['absorption_coef'][p_dict_ind, T_dict_ind]
    return absorb_coef_all_nu[:, nu_dict_ind]


def optical_depth(p, T, wavenumber, molecule_names, q_funcs=None):
    """
    Performs integral of dtau/dp = kq/g over pressure to give optical depth at each pressure level.

    :param p: numpy array [np]
        assume p[0] is smallest pressure i.e. pressure is increasing.
        p must cover whole grid as these pressure values are used for the integral.
    :param T: numpy array [np]
        T[i] is the temperature at pressure level p[i].
    :param wavenumber: numpy array [n_wavenumber]
    :param molecule_names: list [n_molecules]
    :param q_funcs: dictionary of functions, optional.
        q[molecule_name](p) returns the specific humidity of molecule_name at pressure p.
        default: None, meaning Earth values used.
    :return:
        tau: numpy array [np x n_wavenumber]
        tau[-1, :] is surface value
    """
    p = np.sort(p) # ensure pressure is ascending
    tau_integrand = np.zeros((np.size(p), np.size(wavenumber)))

    for molecule_name in molecule_names:
        absorb_coef = np.zeros_like(tau_integrand)
        absorb_coef_dict = np.load(LookupTableFolder + molecule_name + '.npy',
                                                     allow_pickle='TRUE').item()
        update_wavenumber_ind = np.where(np.logical_and(wavenumber >= absorb_coef_dict['nu'].min(),
                                                        wavenumber <= absorb_coef_dict['nu'].max()))[0]
        wavenumber_crop = wavenumber[update_wavenumber_ind]
        absorb_coef[:, update_wavenumber_ind] = get_absorption_coef(p, T, wavenumber_crop, absorb_coef_dict)
        if q_funcs is None:
            q = molecules[molecule_name]['humidity'](p)
        else:
            q = q_funcs[molecule_name](p)
        tau_integrand += absorb_coef * q.reshape(-1, 1)
    tau = np.zeros_like(tau_integrand)
    tau_integrand = tau_integrand / g
    # integrate from p = 0 to p to find tau at p so add p=0 and tau=0 as first row
    tau_integrand = np.concatenate((np.zeros_like(tau_integrand[0, :]).reshape(1, -1), tau_integrand))
    p_integral = np.concatenate(([0], p))
    for i in range(len(p)):
        tau[i, :] = np.trapz(tau_integrand[:i+2], p_integral[:i+2], axis=0)
    return tau


class RealGas(GreyGas):
    def __init__(self, nz, ny, molecule_names, T_g, q_funcs = None,
                 d_nu=10, n_nu_bands=20, T_star=T_sun, R_star=R_sun, star_planet_dist=AU, albedo=0.3):
        self.nz = nz
        self.ny = ny
        self.molecule_names = molecule_names
        if q_funcs is None:
            q_funcs = {}
            for molecule_name in molecule_names:
                q_funcs[molecule_name] = molecules[molecule_name]['humidity']
        self.q_funcs = q_funcs
        self.T_g = T_g
        self.albedo = albedo
        self.star = {'T': T_star, 'R': R_star, 'star_planet_dist': star_planet_dist}
        self.d_nu = d_nu
        self.n_nu_bands = 20
        self.nu, self.nu_lw, self.nu_overlap, self.nu_sw = self.get_wavenumber_array()
        self.bands_lw, self.bands_overlap, self.bands_sw = self.get_wavenumber_bands()
        self.p_interface = self.get_p_grid()
        self.p = np.zeros((self.nz - 1, self.ny))  # at same height as temperature
        for i in range(self.nz - 1):
            self.p[i, :] = np.mean(self.p_interface[i:i + 2, :], 0)
        self.T0 = get_isothermal_temp(albedo=self.albedo, T_star=self.star['T'],
                                      R_star=self.star['R'], star_planet_dist=self.star['star_planet_dist'])
        self.T = np.ones_like(self.p) * self.T0

    def get_wavenumber_array(self, fract_to_ignore=0.001, fract_to_ignore_overlap=0.001):
        nu_initial = np.arange(10.0, 100000.0 + self.d_nu, self.d_nu)
        B_star = B_wavenumber(nu_initial, self.star['T'])
        B_planet = B_wavenumber(nu_initial, self.T_g)
        # only go to wavenumber that accoutns for 99.9% of energy
        max_nu = nu_initial[np.abs(np.cumsum(B_star)/sum(B_star) - (1-fract_to_ignore)).argmin()]
        # start from wavenumber such that only 0.1% of flux neglected
        min_nu = nu_initial[np.abs(np.cumsum(B_planet)/sum(B_planet) - fract_to_ignore).argmin()]

        # get shortwave range which is where solar spectrum dominates planetary spectrum
        # can neglect solar spectrum below sw_nu_min
        sw_nu_min = nu_initial[np.abs(np.cumsum(B_star) / sum(B_star) - fract_to_ignore_overlap).argmin()]
        # can neglect planetary spectrum above lw_nu_max
        lw_nu_max = nu_initial[np.abs(np.cumsum(B_planet)/sum(B_planet) - (1-fract_to_ignore_overlap)).argmin()]
        # in between must consider both
        nu = np.arange(min_nu, max_nu + self.d_nu, self.d_nu)
        nu_overlap = nu[np.logical_and(nu <= lw_nu_max, nu >= sw_nu_min)]
        if len(nu_overlap) == 0:
            sw_nu_min = lw_nu_max
        nu_lw = nu[nu <= sw_nu_min]
        nu_sw = nu[nu >= lw_nu_max]
        return nu, nu_lw, nu_overlap, nu_sw

    def get_wavenumber_bands(self):
        """
        bands small enough that B essentially constant over range of band.
        :return:
        """
        B_star = B_wavenumber(self.nu_sw, self.star['T'])
        # get incoming flux at toa fro star Wm^-2
        # B_toa = B_star * self.star['R']**2 / self.star['star_planet_dist']**2 * (1-self.albedo) / 4
        B_planet = B_wavenumber(self.nu_lw, self.T_g)

        def get_equal_bands(nu, B, n_bands):
            B_norm = B / max(B)
            # turn decrease after peak into increase
            B_norm[B_norm.argmax():] = 1 + (1 - B_norm[B_norm.argmax():])
            B_norm = B_norm - min(B_norm) # make first value always zero
            B_norm = B_norm / max(B_norm)
            target_values = np.linspace(0, 1, n_bands+1)[1:]
            band_info = {'range': [], 'centre': np.zeros(len(target_values)), 'delta': np.zeros(len(target_values))}
            band_start_ind = 0
            for i in range(len(target_values)):
                band_end_ind = max(np.abs(B_norm-target_values[i]).argmin(), band_start_ind+1)
                band_info['range'].append(nu[band_start_ind:band_end_ind+1])
                band_info['centre'][i] = band_info['range'][i][round((len(band_info['range'][i]) + 1) / 2) - 1]
                band_info['delta'][i] = band_info['range'][i][-1] - band_info['range'][i][0]
                band_start_ind = band_end_ind
            return band_info

        B_overlap_planet = B_wavenumber(self.nu_overlap, self.T_g)
        B_overlap_star = B_wavenumber(self.nu_overlap, self.star['T'])
        n_lw_bands = np.sum(B_planet)/(np.sum(B_planet)+np.sum(B_overlap_planet)) * self.n_nu_bands
        n_sw_bands = np.sum(B_star)/(np.sum(B_star)+np.sum(B_overlap_star)) * self.n_nu_bands
        n_overlap_bands = self.n_nu_bands - floor(n_lw_bands) + self.n_nu_bands - floor(n_sw_bands)
        n_lw_bands = ceil(n_lw_bands)
        n_sw_bands = ceil(n_sw_bands)
        bands_lw = get_equal_bands(self.nu_lw, B_planet, n_lw_bands)
        bands_sw = get_equal_bands(self.nu_sw, B_star, n_sw_bands)

        B_overlap_planet = B_overlap_planet / max(B_planet)
        B_overlap_star = B_overlap_star / max(B_star)
        if max(B_overlap_star) == 1 or max(B_overlap_planet) == 1:
            raise ValueError('Peak of planet or star spectrum is in overlap region')
        B_overlap = B_overlap_planet + B_overlap_star[0] - (B_overlap_star-B_overlap_star[0])
        bands_overlap = get_equal_bands(self.nu_overlap, B_overlap, n_overlap_bands)

        """ # Debugging plots
        plt.plot(self.nu_lw, B_planet/max(B_planet))
        plt.scatter(bands_lw['centre'], B_wavenumber(bands_lw['centre'], self.T_g)/max(B_planet))
        plt.plot(self.nu_overlap, B_overlap_planet)
        plt.scatter(bands_overlap['centre'], B_wavenumber(bands_overlap['centre'], self.T_g)/max(B_planet))
        plt.plot(self.nu_overlap, B_overlap_star)
        plt.scatter(bands_overlap['centre'], B_wavenumber(bands_overlap['centre'], self.star['T'])/max(B_star))
        plt.plot(self.nu_sw, B_star/max(B_star))
        plt.scatter(bands_sw['centre'], B_wavenumber(bands_sw['centre'], self.star['T'])/max(B_star))
        plt.show()
        """
        return bands_lw, bands_overlap, bands_sw

    def get_p_grid(self, nz_multiplier_param=100, q_thresh_info_percentile=75,
                   q_thresh_info_max=10, log_p_min_sep=0.1, min_absorb_coef_use=10e-6):
        """
        Get pressure and optical depth grids together so have good separation in both.

        :param nz_multiplier_param: float, optional.
            Each local maxima in mass concentration, q_max, will have nz_multiplier_param*q_max grid points
            associated with it if nz is 'auto'.
            default: 100
        :param q_thresh_info_percentile: float, optional.
            grid around local maxima, q_max runs until q falls to q_max/q_thresh_info_max or to the
            q_thresh_info_percentile of all q.
            default: 75
        :param q_thresh_info_max: float, optional.
            default: 1000
        :param log_p_min_sep: float, optional.
            Try to have a minimum log10 pressure separation of log_p_min_sep in grid.
            default: 0.1
        :param tau_min_sep: float, optional.
            Require a separation in lw optical depth of more than tau_min_sep between pressure levels.
            default: 1e-3
        :return:
        p_interface: numpy array.
            Pressure grid levels for flux calc.
        tau_interface: numpy array.
            Corresponding long wave optical depth
        """
        '''Get base pressure grid in log space'''
        if self.nz == 'auto':
            p_initial_size = int(1e6)
        else:
            p_initial_size = int(self.nz * 1000)
        p_interface = np.logspace(np.log10(p_surface), np.log10(p_toa), p_initial_size)  # can only do with ny=1
        p0 = p_interface.copy()

        '''Get mass concentrations x absorption coef (i.e. dtau/dp) so ensure have grid points around local maxima'''
        q = np.zeros_like(p_interface)
        small = 1e-10
        q_local_maxima_index = []
        for molecule_name in self.molecule_names:
            absorb_coef_dict = np.load(LookupTableFolder + molecule_name + '.npy',
                                       allow_pickle='TRUE').item()
            absorb_coef_all_nu = get_absorption_coef(absorb_coef_dict['p'], np.ones_like(absorb_coef_dict['p']) *
                                                     self.T_g, absorb_coef_dict['nu'], absorb_coef_dict)
            # get absorption coef summed over all significant freq. Each freq contributes same as divide by max
            use_nu = np.max(absorb_coef_all_nu, axis=0) > min_absorb_coef_use
            absorb_coef = np.sum(absorb_coef_all_nu[:, use_nu]/np.max(absorb_coef_all_nu[:, use_nu], axis=0), axis=1)
            coef_interp = interp1d(absorb_coef_dict['p'], absorb_coef)
            absorb_coef = coef_interp(p_interface)
            # to ensure find local maxima if at surface, add value one index away from surface below surface too.
            # subtract small amount to correct for case where value at surface = value one away from surface
            q_molecule = self.q_funcs[molecule_name](p_interface)
            q_molecule = q_molecule/max(q_molecule) * absorb_coef/max(absorb_coef)
            q_molecule_local_maxima_index = argrelextrema(np.insert(q_molecule, 0, q_molecule[1] - small),
                                                          np.greater)[0] - 1
            q_local_maxima_index.append(q_molecule_local_maxima_index[q_molecule_local_maxima_index >= 0])
            q = q + q_molecule
        q = q / len(self.molecule_names)
        cum_q = np.cumsum(q)
        q_local_maxima_index = np.sort(np.concatenate(tuple(q_local_maxima_index)))
        '''Determine number of grid points around each local maxima'''
        last_above_ind = 0  # use max (new_below_ind, above_ind) so local maxima sections don't overlap
        nLocalMaxima = len(q_local_maxima_index)
        q_max_values = q[q_local_maxima_index]
        if self.nz == 'auto':
            nz_multiplier = max(
                [nz_multiplier_param, max(5 / q_max_values)])  # have at least 5 grid points for each local maxima
            nPointsPerSet = np.ceil(q_max_values * nz_multiplier).astype(int)
            self.nz = sum(nPointsPerSet)
        else:
            nz_multiplier = None
            nPointsPerSet = np.floor(q_max_values / sum(q_max_values) * self.nz).astype(int)
            nPointsPerSet[-1] = self.nz - sum(nPointsPerSet[:-1])
        p_array_indices = []
        for i in range(nLocalMaxima):
            if nPointsPerSet[i] > 0:
                # Determine where q falls significantly below local maxima.
                q_thresh = np.min([np.percentile(q, q_thresh_info_percentile),
                                   q[q_local_maxima_index[i]] / q_thresh_info_max])
                if q_local_maxima_index[i] == 0:
                    below_ind = 0
                else:
                    q_below_ind = np.arange(q_local_maxima_index[i])
                    below_ind = max(q_below_ind[np.abs(q[q_below_ind] - q_thresh).argmin()], last_above_ind)
                q_above_ind = np.arange(q_local_maxima_index[i], p_initial_size)
                above_ind = q_above_ind[np.abs(q[q_above_ind] - q_thresh).argmin()]
                # Deal with case where grids around local maxima overlap
                for j in range(i, nLocalMaxima - 1):
                    if above_ind > q_local_maxima_index[j + 1]:
                        nPointsPerSet[i] = nPointsPerSet[i] + nPointsPerSet[j + 1]
                        nPointsPerSet[j + 1] = 0
                if i == 0 and below_ind != 0:
                    nPointsPerSet[i] = nPointsPerSet[i] - 1
                    p_array_indices.append(0)
                if i == nLocalMaxima - 1 and above_ind != p_initial_size - 1:
                    nPointsPerSet[i] = nPointsPerSet[i] - 1
                q_grid_values = np.linspace(cum_q[below_ind], cum_q[above_ind], nPointsPerSet[i])
                p_array_indices_set = [np.abs(cum_q - i).argmin() for i in q_grid_values]
                p_array_indices = p_array_indices + p_array_indices_set
                if i == nLocalMaxima - 1 and above_ind != p_initial_size - 1:
                    p_array_indices.append(p_initial_size - 1)
                last_above_ind = p_array_indices_set[-1] * 2 - p_array_indices_set[
                    -2]  # Set to a step higher than max index
        '''Deal with case where grid is too sparse in pressure space'''
        p_interface = p_interface[p_array_indices]
        log_p = np.log10(p_interface)
        delta_log_p = abs(np.ediff1d(log_p))
        to_correct = np.where(delta_log_p > log_p_min_sep)[0]
        target_log_delta_p = log_p_min_sep / 2
        for i in to_correct:
            if nz_multiplier is not None:
                p_range = np.logical_and(p0 < p_interface[i], p0 > p_interface[i + 1])
                n_new_levels = max(int(max(q[p_range]) * nz_multiplier), 3)
                new_levels = np.logspace(log_p[i], log_p[i + 1], n_new_levels + 2)
                p_interface = np.flip(np.sort(np.append(p_interface, new_levels[1:-1])))
                self.nz = len(p_interface)
            else:
                n_new_levels = int(min(max(np.ceil((log_p[i - 1] - log_p[i]) / target_log_delta_p), 3), self.nz / 10))
                max_i_to_change = int(min(i + np.ceil(n_new_levels / 2), self.nz) - 1)
                min_i_to_change = int(max(max_i_to_change - n_new_levels, 0))
                if min_i_to_change == 0:
                    max_i_to_change = n_new_levels
                new_levels = np.logspace(log_p[min_i_to_change], log_p[max_i_to_change], n_new_levels + 1)
                p_interface[min_i_to_change:max_i_to_change + 1] = new_levels

        p_interface = np.tile(p_interface, (self.ny, 1)).transpose()  # nz x ny
        return p_interface
