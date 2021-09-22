from .real_gas_data.hitran import LookupTableFolder
from ..constants import h_planck, speed_of_light, k_boltzmann, g, T_sun, R_sun, AU, p_surface_earth, p_toa_earth, sigma
from .real_gas_data.specific_humidity import molecules
from .base import Atmosphere, round_any
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy import optimize


def B_freq(freq, T):
    """
    Planck function in terms of frequency (s^-1)
    sigmaT^4 = integral(pi * B_freq) dfreq

    :param freq:
        unit = s^-1
    :param T:
    :return:
    """
    # Large = 1000
    # u = min(h_planck * freq / (k_boltzmann * T), Large) # To prevent overflow
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


def get_absorption_coef(p, T, nu, absorb_coef_dict):
    """
    Given the absorption coefficient data at all pressures, temperatures and wavenumbers
    This returns the value at the subset of pressures and temperatures indicated by p, T and nu.

    :param p: numpy array [np_crop]
    :param T: numpy array [np_crop]
    :param nu: numpy array [n_wavenumber]
    :param absorb_coef_dict: dictionary ['p [np]', 'T [nT]', 'nu [n_nu]', 'absorption_coef [np x nT x n_nu]']
    :return:
        absorb_coef [np_crop x n_wavenumber]
    """
    p_dict_ind = np.abs(p.reshape(-1, 1) - absorb_coef_dict['p'].reshape(-1, 1).transpose()).argmin(axis=1)
    T_dict_ind = np.abs(T.reshape(-1, 1) - absorb_coef_dict['T'].reshape(-1, 1).transpose()).argmin(axis=1)
    nu_dict_ind = np.abs(nu.reshape(-1, 1) -
                         absorb_coef_dict['nu'].reshape(-1, 1).transpose()).argmin(axis=1)
    absorb_coef_all_nu = absorb_coef_dict['absorption_coef'][p_dict_ind, T_dict_ind]
    return absorb_coef_all_nu[:, nu_dict_ind]


def optical_depth(p, T, wavenumber, molecule_names, q_funcs, q_funcs_args):
    """
    Performs integral of dtau/dp = kq/g over pressure to give optical depth at each pressure level.

    :param p: numpy array [np]
        assume p[0] is smallest pressure i.e. pressure is increasing.
        p must cover whole grid as these pressure values are used for the integral.
    :param T: numpy array [np]
        T[i] is the temperature at pressure level p[i].
    :param wavenumber: numpy array [n_wavenumber]
    :param molecule_names: list [n_molecules]
    :param q_funcs: dictionary of functions.
        q_funcs[molecule_name](p, *q_funcs_args[molecule_name])
        returns the specific humidity of molecule_name at pressure p.
    :param q_funcs_args: dictionary of tuples
        q_funcs_args[molecule_name] is arguments to pass to q function as well as pressure.
    :return:
        tau: numpy array [np x n_wavenumber]
        tau[-1, :] is surface value
    """
    p = np.sort(p)  # ensure pressure is ascending
    tau_integrand = np.zeros((np.size(p), np.size(wavenumber)))

    for molecule_name in molecule_names:
        absorb_coef = np.zeros_like(tau_integrand)
        absorb_coef_dict = np.load(LookupTableFolder + molecule_name + '.npy',
                                   allow_pickle='TRUE').item()
        update_wavenumber_ind = np.where(np.logical_and(wavenumber >= absorb_coef_dict['nu'].min(),
                                                        wavenumber <= absorb_coef_dict['nu'].max()))[0]
        wavenumber_crop = wavenumber[update_wavenumber_ind]
        absorb_coef[:, update_wavenumber_ind] = get_absorption_coef(p, T, wavenumber_crop, absorb_coef_dict)
        q = q_funcs[molecule_name](p, *q_funcs_args[molecule_name])
        tau_integrand += absorb_coef * q.reshape(-1, 1)
    tau = np.zeros_like(tau_integrand)
    tau_integrand = tau_integrand / g
    # integrate from p = 0 to p to find tau at p so add p=0 and tau=0 as first row
    tau_integrand = np.concatenate((np.zeros_like(tau_integrand[0, :]).reshape(1, -1), tau_integrand))
    p_integral = np.concatenate(([0], p))
    for i in range(len(p)):
        tau[i, :] = np.trapz(tau_integrand[:i + 2], p_integral[:i + 2], axis=0)
    return tau


def transmission(p1, p2, p_all, nu_band, delta_nu_band, nu_all, tau):
    """
    Computes the transmission function for a given wavenumber band.

    :param p1: numpy array [np1]
    :param p2: numpy array [np2]
    :param p_all: numpy array [np]
    :param nu_band: numpy array [n_band]
    :param delta_nu_band: float
    :param nu_all: numpy array [n_nu]
    :param tau: numpy array [np x n_nu]
    :return: numpy array [np1 x np2]
    """
    p1_ind = np.where(p_all.reshape(-1, 1) == p1)[0]
    p2_ind = np.where(p_all.reshape(-1, 1) == p2)[0]
    nu_ind = np.where(nu_all.reshape(-1, 1) == nu_band)[0]
    if len(p1_ind) == 0 or len(p2_ind) == 0 or len(nu_ind) == 0:
        raise ValueError('p1, p2 or nu have no values in p_all or nu_all')
    tau = tau[:, nu_ind]
    tau_p1 = np.expand_dims(tau[p1_ind, :], axis=1)
    tau_p1 = np.repeat(tau_p1, len(p2), axis=1)
    tau_p2 = np.expand_dims(tau[p2_ind, :], axis=0)
    tau_p2 = np.repeat(tau_p2, len(p1), axis=0)
    integrand = np.exp(tau_p1 - tau_p2)
    return np.trapz(integrand, nu_band, axis=2) / delta_nu_band


class RealGas(Atmosphere):
    def __init__(self, nz, ny, molecule_names, T_g=None, q_funcs=None, q_funcs_args=None, n_nu_bands=40,
                 T_star=T_sun, R_star=R_sun, star_planet_dist=AU, albedo=0.3, temp_change=1, T_func=None,
                 p_surface=p_surface_earth, p_toa=p_toa_earth):
        """

        :param nz: integer or 'auto'.
            Number of pressure levels. If 'auto', chooses an appropriate amount such that have good spread
            in both pressure and optical thickness, tau
        :param ny: integer
            Number of latitudes
        :param molecule_names: list of strings
            Will find contribution to optical depth from each of these molecules e.g. 'CO2', 'H2O'
        :param T_g: float, optional.
            Surface temperature in K. If None, will guess value to give approximate balance between fluxes at
            top of atmosphere.
            default: None
        :param q_funcs: dictionary of functions
            q_funcs[molecule_name](p, *q_funcs_args[molecule_name]) would compute the specific humidity due to
            molecule_name at pressure p. If None, typical earth functions used.
            default: None
        :param q_funcs_args: dictionary of tuples
            arguments to q_funcs other than pressure.
            default: None
        :param n_nu_bands: integer, optional.
            number of wavenumber bands to use in flux calculations.
            default: 40
        :param T_star: float, optional.
            Equivalent black body temperature of the starin K
            default: T_sun
        :param R_star: float, optional.
            Radius of star (m).
            default: R_sun
        :param star_planet_dist: float, optional.
            Distance between star and planet (m)
            default: 1 AU
        :param albedo: list, optional.
            Upward short wave flux at top of atmosphere is (1-albedo[i])*F_stellar_constant*latitude_dist_factor[i]/4;
            at each latitude index i. If give single value, will repeat this value at each latitude.
            Can also be function of latitude.
            default: 0.3 at each latitude.
        :param temp_change: float, optional.
            Time step is found so that at least one level will have a temperature change equal to this.
            default: 1K.
        :param T_func: function, optional.
            Function to compute temperature profile given pressure as only argument.
            Useful if want to compute OLR for specific Temperature profile.
            If None, set to isothermal temperature everywhere.
            default: None
        :param p_surface: float, optional.
            Pressure in Pa at surface.
            default: p_surface_earth = 101320 Pa
        :param p_toa: float, optional.
            Pressure in Pa at top of atmosphere
            default: p_toa_earth = 20 Pa
        """
        self.star = {'T': T_star, 'R': R_star, 'star_planet_dist': star_planet_dist}
        F_stellar_constant = sigma * self.star['T'] ** 4 * self.star['R'] ** 2 / \
                             self.star['star_planet_dist'] ** 2
        super().__init__(nz, ny, F_stellar_constant, albedo, p_surface, p_toa, temp_change)
        if T_g is None:
            # assume some greenhouse warming (Guess of ground temperature. Needed to work out pressure grid)
            self.T_g = self.T0 + 20
        else:
            self.T_g = T_g
        self.molecule_names = molecule_names
        if q_funcs is None:
            q_funcs = {}
            q_funcs_args = {}
            for molecule_name in molecule_names:
                q_funcs[molecule_name] = molecules[molecule_name]['q']
                q_funcs_args[molecule_name] = molecules[molecule_name]['q_args']
        self.q_funcs = q_funcs
        self.q_funcs_args = q_funcs_args
        # get wavenumber spacing from absorption coefficient data. Assume same for all molecules
        absorb_coef_nu = np.load(LookupTableFolder + molecule_names[0] + '.npy',
                                 allow_pickle='TRUE').item()['nu']
        self.d_nu = absorb_coef_nu[1] - absorb_coef_nu[0]
        self.n_nu_bands = n_nu_bands
        self.nu, self.nu_lw, nu_overlap, self.nu_sw = self.get_wavenumber_array()
        self.nu_bands = self.get_wavenumber_bands(nu_overlap)
        self.p_interface = np.sort(self.get_p_grid(), axis=0)  # ascending
        self.p = np.zeros((self.nz - 1, self.ny))  # at same height as temperature
        for i in range(self.nz - 1):
            self.p[i, :] = np.mean(self.p_interface[i:i + 2, :], 0)
        if T_func is None:
            self.T = np.ones_like(self.p) * self.T0
            self.T_interface = np.ones_like(self.p_interface[:, 0]) * self.T0
        else:
            self.T = T_func(self.p)
            self.T_interface = T_func(self.p_interface[:, 0])
        self.tau_interface = optical_depth(self.p_interface[:, 0], self.T_interface, self.nu,
                                           self.molecule_names, self.q_funcs, self.q_funcs_args)
        self.up_flux, self.down_flux = self.get_flux()
        self.net_flux = np.sum(self.up_flux * self.nu_bands['delta'], axis=1) - \
                        np.sum(self.down_flux * self.nu_bands['delta'], axis=1)
        if T_g is None:
            # update guess of ground temperature
            self.inital_Tg_guess()
        delattr(self, 'T_interface') # only needed for tau calculation.

    def get_wavenumber_array(self, fract_to_ignore=0.001, fract_to_ignore_overlap=0.001):
        """
        Gets wavenumber values which cover both planetary and stellar spectrum.

        :param fract_to_ignore: float, optional.
            Wavenumber range is chosen so that this fraction of planetary and stellar flux is neglected.
            default: 0.001
        :param fract_to_ignore_overlap: float, optional.
            fraction of planetary flux over which we can neglect integral in flux calculation.
        :return:
            nu: all wavenumbers [n_nu]
            nu_lw: wavenumbers over which we need to perform the integral in flux calc [n_nu_lw]
            nu_overlap: wavenumbers over which we must consider stellar and planetary spectrums when finding
                wavenumber bands [n_nu_overlap]
            nu_sw: wavenumbers over which we can ignore integral in flux calc [n_nu - n_nu_lw]
        """
        nu_initial = np.arange(10.0, 100000.0 + self.d_nu, self.d_nu)
        B_star = B_wavenumber(nu_initial, self.star['T'])
        B_planet = B_wavenumber(nu_initial, self.T_g)
        # only go to wavenumber that accoutns for 99.9% of energy
        max_nu = nu_initial[np.abs(np.cumsum(B_star) / sum(B_star) - (1 - fract_to_ignore)).argmin()]
        # start from wavenumber such that only 0.1% of flux neglected
        min_nu = nu_initial[np.abs(np.cumsum(B_planet) / sum(B_planet) - fract_to_ignore).argmin()]

        # get shortwave range which is where solar spectrum dominates planetary spectrum
        # can neglect solar spectrum below sw_nu_min
        sw_nu_min = nu_initial[np.abs(np.cumsum(B_star) / sum(B_star) - fract_to_ignore_overlap).argmin()]
        # can neglect planetary spectrum above lw_nu_max
        lw_nu_max = nu_initial[np.abs(np.cumsum(B_planet) / sum(B_planet) - (1 - fract_to_ignore_overlap)).argmin()]
        # in between must consider both
        nu = np.arange(min_nu, max_nu + self.d_nu, self.d_nu)
        nu_overlap = nu[np.logical_and(nu <= lw_nu_max, nu >= sw_nu_min)]
        nu_lw = nu[nu <= lw_nu_max]
        nu_sw = nu[nu >= lw_nu_max]
        return nu, nu_lw, nu_overlap, nu_sw

    def get_wavenumber_bands(self, nu_overlap):
        """
        Gets self.n_nu_bands in way that best keeps planetary+stellar flux constant over each wavenumber band.
        :return: dictonary ['range': lists, 'centre': floats, 'delta': floats, 'sw': booleans]
        """
        B_star = B_wavenumber(self.nu_sw, self.star['T'])
        # get incoming flux at toa fro star Wm^-2
        # B_toa = B_star * self.star['R']**2 / self.star['star_planet_dist']**2 * (1-self.albedo) / 4
        nu_lw_only = np.setdiff1d(self.nu_lw, nu_overlap)
        B_planet = B_wavenumber(nu_lw_only, self.T_g)

        def get_equal_bands(nu, B, n_bands):
            B_norm = B / max(B)
            # turn decrease after peak into increase
            B_norm[B_norm.argmax():] = 1 + (1 - B_norm[B_norm.argmax():])
            B_norm = B_norm - min(B_norm)  # make first value always zero
            B_norm = B_norm / max(B_norm)
            target_values = np.linspace(0, 1, n_bands + 1)[1:]
            band_info = {'range': [], 'centre': np.zeros(len(target_values)), 'delta': np.zeros(len(target_values))}
            band_start_ind = 0
            for i in range(len(target_values)):
                band_end_ind = max(np.abs(B_norm - target_values[i]).argmin(), band_start_ind + 1)
                band_info['range'].append(nu[band_start_ind:band_end_ind + 1])
                band_info['centre'][i] = band_info['range'][i][round((len(band_info['range'][i]) + 1) / 2) - 1]
                band_info['delta'][i] = band_info['range'][i][-1] - band_info['range'][i][0]
                band_start_ind = band_end_ind
            return band_info

        B_overlap_planet = B_wavenumber(nu_overlap, self.T_g)
        B_overlap_star = B_wavenumber(nu_overlap, self.star['T'])
        n_planet_overlap = (1 - np.sum(B_planet) / (np.sum(B_planet) + np.sum(B_overlap_planet))) * self.n_nu_bands / 2
        n_star_overlap = (1 - (np.sum(B_star) / (np.sum(B_star) + np.sum(B_overlap_star)))) * self.n_nu_bands / 2
        n_overlap_bands = ceil(n_planet_overlap + n_star_overlap)
        n_lw_bands = ceil(self.n_nu_bands / 2 - n_planet_overlap)
        n_sw_bands = self.n_nu_bands - n_lw_bands - n_overlap_bands
        bands_lw = get_equal_bands(nu_lw_only, B_planet, n_lw_bands)
        bands_sw = get_equal_bands(self.nu_sw, B_star, n_sw_bands)

        # in overlap region must consider both stellar and planetary flux.
        # Here we add them together in such a way that flux increases with wavenumber.
        B_overlap_planet = B_overlap_planet / max(B_planet)
        B_overlap_star = B_overlap_star / max(B_star)
        if max(B_overlap_star) == 1 or max(B_overlap_planet) == 1:
            raise ValueError('Peak of planet or star spectrum is in overlap region')
        B_overlap = B_overlap_planet + B_overlap_star[0] - (B_overlap_star - B_overlap_star[0])
        bands_overlap = get_equal_bands(nu_overlap, B_overlap, n_overlap_bands)

        # Combine bands to one dict, include 'sw'. If 'sw' = True, dont have to compute integral in flux calc
        bands = {'range': bands_lw['range'] + bands_overlap['range'] + bands_sw['range'],
                 'centre': np.concatenate((bands_lw['centre'], bands_overlap['centre'], bands_sw['centre'])),
                 'delta': np.concatenate((bands_lw['delta'], bands_overlap['delta'], bands_sw['delta'])),
                 'sw': np.ones(self.n_nu_bands, dtype=bool)}
        bands['sw'][bands['centre'] <= self.nu_sw.min()] = False

        """ # Debugging plots
        plt.plot(nu_lw_only, B_planet / max(B_planet))
        plt.scatter(bands_lw['centre'], B_wavenumber(bands_lw['centre'], self.T_g) / max(B_planet))
        plt.plot(nu_overlap, B_overlap_planet)
        plt.scatter(bands_overlap['centre'], B_wavenumber(bands_overlap['centre'], self.T_g) / max(B_planet))
        plt.plot(nu_overlap, B_overlap_star)
        plt.scatter(bands_overlap['centre'], B_wavenumber(bands_overlap['centre'], self.star['T']) / max(B_star))
        plt.plot(self.nu_sw, B_star / max(B_star))
        plt.scatter(bands_sw['centre'], B_wavenumber(bands_sw['centre'], self.star['T']) / max(B_star))
        # plt.xlim(0,5000)
        plt.show()
        """
        return bands

    def get_p_grid(self, min_absorb_coef_use=10e-6, min_log_p_spacing_factor=5000, max_log_p_spacing_factor=50,
                   max_max_log_p_spacing=0.2):
        """
        Get pressure grid so grid points are most dense where dtau/dp or the specific humidity x absorption coefficient
        is largest. This is so there is more resolution in areas where the atmosphere affects radiation more.

        :param min_absorb_coef_use: float, optional
            used to get absorption coef summed over all significant nu. significant nu where absorbption
            coefficient greater than this.
            default: 10^-6
        :param min_log_p_spacing_factor: float, optional
            The minimum spacing of log pressure is -log(q_max)/min_log_p_spacing_factor
            default: 5000
        :param max_log_p_spacing_factor: float, optional
            The maximum spacing of log pressure is -log(q_min)/max_log_p_spacing_factor
            default: 50
        :param max_max_log_p_spacing: float, optional
            any spacing of log pressure cannot exceed this.
            default: 0.2
        :return:
        p_interface: numpy array.
            Pressure grid levels for flux calc.
        """
        '''Get base pressure grid in log space'''
        if self.nz == 'auto':
            p_initial_size = int(1e6)
        else:
            p_initial_size = int(self.nz * 1000)
        p_interface = np.logspace(np.log10(self.p_surface), np.log10(self.p_toa), p_initial_size)  # can only do with ny=1

        '''Get mass concentrations x absorption coef (i.e. dtau/dp) so ensure have grid points around local maxima'''
        q = np.zeros_like(p_interface)
        for molecule_name in self.molecule_names:
            absorb_coef_dict = np.load(LookupTableFolder + molecule_name + '.npy',
                                       allow_pickle='TRUE').item()
            absorb_coef_all_nu = get_absorption_coef(absorb_coef_dict['p'], np.ones_like(absorb_coef_dict['p']) *
                                                     self.T_g, absorb_coef_dict['nu'], absorb_coef_dict)
            # get absorption coef summed over all significant freq.
            # just use absorb_coef to modify q relative to max pressure value
            use_nu = np.max(absorb_coef_all_nu, axis=0) > min_absorb_coef_use
            absorb_coef = np.mean(absorb_coef_all_nu[:, use_nu], axis=1)
            absorb_coef = absorb_coef / np.max(absorb_coef)
            if len(absorb_coef) > 1:
                coef_interp = interp1d(absorb_coef_dict['p'], absorb_coef)
                to_interp = np.where(p_interface >= absorb_coef_dict['p'].min())[0]
                absorb_coef = np.ones_like(p_interface)
                absorb_coef[to_interp] = coef_interp(p_interface[to_interp])
                # set low pressure values to smallest pressure value in data table.
                absorb_coef[p_interface < absorb_coef_dict['p'].min()] = absorb_coef[to_interp[-1]]

            # to ensure find local maxima if at surface, add value one index away from surface below surface too.
            # subtract small amount to correct for case where value at surface = value one away from surface
            q_molecule = self.q_funcs[molecule_name](p_interface, *self.q_funcs_args[molecule_name])
            q = q + q_molecule * absorb_coef

        log_p_array = np.log10(p_interface)
        if self.nz == 'auto':
            # make spacing of log pressure levels dependent on specific humidity, q
            # i.e. larger q means more stuff means we need more pressure levels around it
            log_q_array = np.log10(q)
            log_q_array[q==0] = log_q_array[q > 0].min()  #  ensure no nan values
            min_log_p_spacing = -log_q_array.max() / min_log_p_spacing_factor
            max_log_p_spacing = np.clip(-log_q_array.min() / max_log_p_spacing_factor,
                                        min_log_p_spacing, max_max_log_p_spacing)
            # deal with cases where most of pressure levels are near maxima
            # to avoid nz too large.
            fract_large = sum(q > 0.9 * q.max()) / len(q)
            min_log_p_spacing = fract_large * max_log_p_spacing + (1-fract_large) * min_log_p_spacing

            def get_log_p_spacing(log_q):
                if log_q_array.min() == log_q_array.max():
                    return min_log_p_spacing
                else:
                    gradient = (max_log_p_spacing - min_log_p_spacing) / (log_q_array.min() - log_q_array.max())
                    intercept = max_log_p_spacing - gradient * log_q_array.min()
                    return gradient * log_q + intercept

            log_p_current = log_p_array[0]
            log_p_final = []
            while log_p_current > log_p_array[-1]:
                log_p_final.append(log_p_current)
                ind = np.abs(log_p_array - log_p_current).argmin()
                log_p_current = log_p_final[-1] - get_log_p_spacing(log_q_array[ind])
            # ensure includes surface and toa pressures
            log_p_final = np.array(log_p_final)
            cum_diff = np.cumsum(abs(np.ediff1d(log_p_final)))
            scale_factor = (log_p_array[0] - log_p_array[-1]) / cum_diff[-1]
            cum_diff = cum_diff * scale_factor
            log_p_final = np.concatenate((log_p_final[:1], log_p_final[0] - cum_diff))
            self.nz = len(log_p_final)
        else:
            # log_p_0 = log_p_array[0]
            # log_p_1 = log_p_0 - min_log_p_spacing
            # log_p_n = log_p_array[-1]
            # alpha = np.log(log_p_0 - log_p_1 + 1) / np.log(log_p_0 + 1 - log_p_n) / (1-self.nz)
            # beta = np.exp((np.log(log_p_0 - log_p_1 + 1) / alpha))
            # log_p_final = log_p_0 + 1 - beta ** (alpha * np.arange(self.nz + 1))
            # cover all pressures in manner that log spacing between pressure levels is smaller
            # near the surface
            alpha = np.log10(log_p_array[0] - log_p_array[-1] + 1) / (self.nz - 1)
            log_p_final = log_p_array[0] + 1 - 10 ** (alpha * np.arange(self.nz))
            if log_p_final[-1] != log_p_array[-1]:
                raise ValueError('Too few grid points to cover pressure grid')

        p_interface = 10 ** log_p_final
        return np.tile(p_interface, (self.ny, 1)).transpose()

    def inital_Tg_guess(self):
        """
        Update T_g so sum of net_flux over all pressure levels is initially 0
        """

        def f(x):
            self.T_g = x
            self.up_flux, self.down_flux = self.get_flux()
            self.net_flux = np.sum(self.up_flux * self.nu_bands['delta'], axis=1) - \
                            np.sum(self.down_flux * self.nu_bands['delta'], axis=1)
            return sum(self.net_flux)

        self.T_g = optimize.newton(f, self.T_g)
        # update wavenumber bands given new ground temperature (surface radiation now changed)
        self.nu, self.nu_lw, nu_overlap, self.nu_sw = self.get_wavenumber_array()
        self.nu_bands = self.get_wavenumber_bands(nu_overlap)
        self.tau_interface = optical_depth(self.p_interface[:, 0], self.T_interface, self.nu,
                                           self.molecule_names, self.q_funcs, self.q_funcs_args)
        self.up_flux, self.down_flux = self.get_flux()
        self.net_flux = np.sum(self.up_flux * self.nu_bands['delta'], axis=1) - \
                        np.sum(self.down_flux * self.nu_bands['delta'], axis=1)

    def find_Tg(self, flux_thresh=0.1, tol=0.01, convective_adjust=False):
        """
        finds ground temperature such that net flux at top of atmosphere is approximately less than tol

        :param flux_thresh: float, optional.
            Threshold in delta_net_flux to achieve equilibrium
            default: 0.1
        :param tol: float, optional.
            default: 0.1
        :param convective_adjust: boolean, optional.
            Whether the temperature profile should adjust to stay stable with respect to convection.
            default: False
        :return: T_g
        """
        print("Finding ground temperature to give top of atmosphere flux balance...")

        def f(x):
            try:
                print("Trying T_g = {:.1f} K".format(x))
            except TypeError:
                print("Trying T_g = {:.1f} K".format(x[0]))
            self.T_g = x
            _ = self.evolve_to_equilibrium(flux_thresh=flux_thresh, save=False, convective_adjust=convective_adjust)
            return self.net_flux[0]

        root = optimize.newton(f, self.T_g, tol=tol)
        return root[0]

    def flux_integrals(self, j, T_interface):
        """
        Performs the integrals that appear in the up and down flux equations

        :param j: integer
            wavenumber index between 0 and self.n_nu_bands-1
        :param T_interface: numpy array [nz]
            Need Temperature at interfaces to compute integrand at limits of integral
            Use central cell temperatures and pressures within integral itself.
        :return:
            integral_up, integral_down [nz]
        """
        transmission_array = transmission(self.p_interface[:, 0], self.p_interface[:, 0],
                                          self.p_interface[:, 0], self.nu_bands['range'][j],
                                          self.nu_bands['delta'][j], self.nu,
                                          self.tau_interface)  # [nz x nz]
        """Up"""
        integral_up = np.zeros(self.nz)
        # value of differential at central levels i.e. same levels as T and p
        # found by taking difference between adjacent interface levels i.e. p_interface
        dtransmission_dp_up = np.diff(transmission_array, axis=1) / np.diff(self.p_interface[:, 0])
        interior_integrand_up = np.pi * B_wavenumber(self.nu_bands['centre'][j], self.T)
        interior_integrand_up = dtransmission_dp_up * interior_integrand_up.reshape(-1)  # [nz x nz-1]
        # up flux integral always ends at surface
        surface_integrand_up = np.pi * B_wavenumber(self.nu_bands['centre'][j], self.T_g
                                                    ) * dtransmission_dp_up[:, -1]  # [nz]
        # extremities are interface values, interiors are pressures that align with self.T
        p_integ_range_up = np.concatenate((self.p_interface[:1, 0], self.p[:, 0], self.p_interface[-1:, 0]))

        """Down"""
        integral_down = np.zeros(self.nz)
        dtransmission_dp_down = np.diff(transmission_array, axis=0) / np.diff(self.p_interface[:, 0]).reshape(-1, 1)
        interior_integrand_down = np.pi * B_wavenumber(self.nu_bands['centre'][j], self.T)
        interior_integrand_down = dtransmission_dp_down * interior_integrand_down  # [nz-1 x nz]
        # down flux integral always begins at top of atmosphere
        toa_integrand_down = np.pi * B_wavenumber(self.nu_bands['centre'][j], T_interface[0]
                                                  ) * dtransmission_dp_down[0, :]  # [nz]
        p_integ_range_down = np.concatenate((self.p_interface[:1, 0], self.p[:, 0], self.p_interface[-1:, 0]))

        for i in range(self.nz - 1):
            """Up"""
            interior_use_up = self.p[:, 0] > self.p_interface[i, 0]
            p_integ_range_up[0] = self.p_interface[i]  # make lower limit is interface value
            lower_interface_integrand_up = np.pi * B_wavenumber(self.nu_bands['centre'][j], T_interface[i]) * \
                                           dtransmission_dp_up[i, i]
            integrand_up = np.concatenate(
                (lower_interface_integrand_up, interior_integrand_up[i, interior_use_up],
                 surface_integrand_up[i:i + 1]))
            integral_up[i] = -np.trapz(integrand_up, p_integ_range_up)
            p_integ_range_up = p_integ_range_up[1:]  # get ready for next level

            """Down"""
            p_integ_range_down[-1] = self.p_interface[self.nz - i - 1]  # make upper limit is interface value
            interior_use_down = self.p[:, 0] < self.p_interface[self.nz - i - 1, 0]
            upper_interface_integrand_down = np.pi * B_wavenumber(self.nu_bands['centre'][j],
                                                                  T_interface[self.nz - i - 1]) * \
                                             dtransmission_dp_down[self.nz - i - 2, self.nz - i - 1]
            integrand_down = np.concatenate(
                (toa_integrand_down[self.nz - i - 1:self.nz - i],
                 interior_integrand_down[interior_use_down, self.nz - i - 1],
                 upper_interface_integrand_down))
            integral_down[self.nz - i - 1] = np.trapz(integrand_down, p_integ_range_down)
            p_integ_range_down = p_integ_range_down[:-1]  # get ready for next level
        return integral_up, integral_down

    def get_flux(self):
        """
        Calculates up and down flux arrays at each pressure and wavenumber [nz x n_nu_bands]
        """
        # need T at interfaces for integrals
        T_interp_func = InterpolatedUnivariateSpline(self.p, self.T)
        T_interface = T_interp_func(self.p_interface)
        T_interface[-1] = self.T_g  # ground temperature at highest pressure interface
        # initialise up flux array with surface flux everywhere
        up_flux = np.ones((self.nz, self.n_nu_bands)) * np.pi * B_wavenumber(self.nu_bands['centre'], self.T_g)
        # initialise down flux array with top of atmosphere flux everywhere
        down_flux = np.ones((self.nz, self.n_nu_bands)) * \
                    np.pi * B_wavenumber(self.nu_bands['centre'], self.star['T']) * \
                    self.star['R'] ** 2 / self.star['star_planet_dist'] ** 2 * (1 - self.albedo) / 4
        for j in range(self.n_nu_bands):
            # Apply exponential decay of surface flux at lower pressures
            up_flux[:, j] = up_flux[:, j] * transmission(self.p_interface[:, 0], self.p_interface[-1:, 0],
                                                         self.p_interface[:, 0], self.nu_bands['range'][j],
                                                         self.nu_bands['delta'][j], self.nu,
                                                         self.tau_interface).reshape(-1)
            # Apply exponential decay of top of atmosphere flux at higher pressures
            down_flux[:, j] = down_flux[:, j] * transmission(self.p_interface[:1, 0], self.p_interface[:, 0],
                                                             self.p_interface[:, 0], self.nu_bands['range'][j],
                                                             self.nu_bands['delta'][j], self.nu,
                                                             self.tau_interface).reshape(-1)
            if not self.nu_bands['sw'][j]:
                # for wavenumbers in long wave region, need to consider emission by atmosphere itself
                # atmosphere does not emit in short wave region hence neglect integral
                integral_up, integral_down = self.flux_integrals(j, T_interface)
                up_flux[:, j] += integral_up
                down_flux[:, j] += integral_down
        return up_flux, down_flux

    def take_time_step(self, t, T_initial=None, changing_tau=False, convective_adjust=False,
                       net_flux_thresh=1e-7, net_flux_percentile=95):
        """
        This finds the fluxes given the current temperature profile. It then finds a suitable time step through
        the function update_time_step. It will then update the temperature profile accordingly.

        :param t: float.
            time since start of simulation.
        :param T_initial: numpy array, optional.
            Temperature profile to start simulation from. If not specified, will use isothermal profile.
            Will only apply if t==0.
            default: None
        :param changing_tau: boolean, optional.
            Whether the optical depth is changing with time. If it is not, algorithm will adjust to try to reach
            equilibrium.
            default: False
        :param convective_adjust: boolean, optional.
            Whether the temperature profile should adjust to stay stable with respect to convection.
            default: False
        :param net_flux_thresh: float, optional.
            Only update temperature at pressure levels where change in net flux between time steps
            is less than net_flux_thresh.
            default: 1e-7.
        :param net_flux_percentile: float, optional.
            delta_net_flux is the change in net flux of the percentile of the grid given by this.
            So 100 means max, lower gives some leeway to outliers.
            default: 95.
        :return:
        t: time after current time step.
        delta_net_flux: difference in net flux between time steps. Lower means reaching convergence.
        """
        if t == 0 and T_initial is not None:
            self.T = T_initial
        self.up_flux, self.down_flux = self.get_flux()
        # sum over wavenumbers
        net_flux = np.sum(self.up_flux * self.nu_bands['delta'], axis=1) - \
                   np.sum(self.down_flux * self.nu_bands['delta'], axis=1)
        # net_lw_flux should be zero everywhere in equilibrium
        t, delta_net_flux = self.update_temp(t, net_flux.reshape(-1, 1), changing_tau, convective_adjust,
                                             net_flux_thresh, net_flux_percentile)
        return t, delta_net_flux

    def save_data(self, data_dict, t):
        """
        This appends the time and current Temperature to the data_dict. It will also add optical depth and
        flux data if they are already in data_dict.

        :param data_dict: dictionary.
            Must contain 't' and 'T' keys. Can also contain 'flux' keys.
            The info in this dictionary is used to pass to plot_animate.
        :param t: float.
            Current time.
        :return:
            data_dict
        """
        data_dict['t'].append(t)
        data_dict['T'].append(self.T.copy())
        if "flux" in data_dict:
            sw = self.nu_bands['sw']
            lw = sw == False
            data_dict['flux']['lw_up'].append(np.sum(self.up_flux[:, lw] * self.nu_bands['delta'][lw], axis=1))
            data_dict['flux']['lw_down'].append(np.sum(self.down_flux[:, lw] * self.nu_bands['delta'][lw], axis=1))
            data_dict['flux']['sw_up'].append(np.sum(self.up_flux[:, sw] * self.nu_bands['delta'][sw], axis=1))
            data_dict['flux']['sw_down'].append(np.sum(self.down_flux[:, sw] * self.nu_bands['delta'][sw], axis=1))
        return data_dict

    def plot_olr(self, olr_label='Top of atmosphere'):
        """
        Plots flux emitted by surface in long wave regime (uses integral in flux calc).
        Also plots upward flux in long wave regime at top of atmosphere (Outgoing longwave radiation OLR).
        :param olr_label: label to give top of atmosphere flux.
        :return: ax so can add new lines.
        """
        surface_up_flux = B_wavenumber(self.nu_lw, self.T_g) * np.pi
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.nu_lw, surface_up_flux, color='k', label='$T_g={:.0f}$K blackbody'.format(self.T_g))
        bands_use = self.nu_bands['sw'] == False
        bands_use[np.where(bands_use == False)[0][0]] = True # add extra band so not cut off before end of axis
        ax.scatter(self.nu_bands['centre'][bands_use],
                   B_wavenumber(self.nu_bands['centre'][bands_use], self.T_g) * np.pi, color='k', s=10)
        ax.plot(self.nu_bands['centre'][bands_use], self.up_flux[0, bands_use], label=olr_label)
        ax.set_xlim((0, round_any(self.nu_lw.max(), 500, 'ceil')))
        ax.set_ylim((0, round_any(surface_up_flux.max(), 0.05, 'ceil')))
        ax.set_xlabel('Wavenumber cm$^{-1}$')
        ax.set_ylabel('Flux Density ((W/m$^2$)/cm$^{-1}$)')
        ax.legend()
        ax.set_title('Upward Planetary Radiation')
        return ax

    def plot_incoming_short_wave(self, sw_label='Surface'):
        """
        Plots incoming radiation at top of atmosphere in short wave regime (no integral in flux calc).
        Also plots downward flux in short wavenumber regime at surface.
        :param sw_label: label to give surface flux.
        :return: ax so can add new lines.
        """
        def solar_flux(nu):
            return B_wavenumber(nu, self.star['T']) * np.pi * \
                   self.star['R'] ** 2 / self.star['star_planet_dist'] ** 2 * (1 - self.albedo) / 4
        toa_flux = solar_flux(self.nu_sw)
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.nu_sw, toa_flux, color='k', label='Top of atmosphere')
        bands_use = self.nu_bands['sw']
        ax.scatter(self.nu_bands['centre'][bands_use],
                   solar_flux(self.nu_bands['centre'][bands_use]), color='k', s=10)
        ax.plot(self.nu_bands['centre'][bands_use], self.down_flux[-1, bands_use], label=sw_label)
        ax.set_xlim((0, round_any(self.nu_sw.max(), 10000, 'ceil')))
        ax.set_ylim((0, round_any(toa_flux.max(), 0.005, 'ceil')))
        ax.set_xlabel('Wavenumber cm$^{-1}$')
        ax.set_ylabel('Flux Density ((W/m$^2$)/cm$^{-1}$)')
        ax.legend()
        ax.set_title('Downward Solar Radiation')
        return ax
