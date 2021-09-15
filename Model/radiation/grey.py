from ..constants import g, c_p_dry, sigma, p_surface, p_toa, F_sun
from .convective_adjustment import convective_adjustment
import Model.radiation.grey_optical_depth as od
import numpy as np
from numba import jit
from sympy import symbols, lambdify, diff, exp, simplify, sympify, integrate, cancel, Function
from inspect import signature
from sympy.solvers import solve
import matplotlib.pyplot as plt
import matplotlib
import warnings
import inspect
from itertools import groupby
from operator import itemgetter
from scipy.signal import argrelextrema
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable


class GreyGas:

    def __init__(self, nz, ny, tau_lw_func, tau_lw_func_args, tau_sw_func=None,
                 tau_sw_func_args=None, F_stellar_constant=F_sun, albedo=0.3):
        """
        Finds grid in pressure and both optical depth (tau) and gas mass concentration (q) for the long wave
        and short wave absorbing gases (lw and sw). Interface values are used to calculate flux and these
        interfaces bound the cell where we get a temperature value for. Initial temperature is set
        to the isothermal solution in absence of atmosphere.

        :param nz: integer or 'auto'.
            Number of pressure levels. If 'auto', chooses an appropriate amount such that have good spread
            in both pressure and optical thickness, tau
        :param tau_lw_func: function.
            Function in grey_optical_depth to calculate long wave optical depth from pressure
        :param tau_lw_func_args: list.
            Arguments to pass to tau_lw_func
        :param tau_sw_func: function, optional.
            Function in grey_optical_depth.py to calculate short wave optical depth from pressure
            If not given, atmosphere will not affect any short wave radiation.
            default: None
        :param tau_sw_func_args: list, optional.
            Arguments to pass to tau_sw_func;
            default: None
        :param F_stellar_constant: float, optional.
            Flux density at surface of planet. Units = W/m^2;
            default: F_solar
        :param albedo: list, optional.
            Upward short wave flux at top of atmosphere is (1-albedo[i])*F_stellar_constant*latitude_dist_factor[i]/4;
            at each latitude index i.
            If give single value, will repeat this value at each latitude.
            default: 0.3 at each latitude.
        """
        self.ny = ny
        self.nz = nz
        self.latitude = np.linspace(-90, 90, self.ny)
        if inspect.isfunction(albedo):
            self.albedo = albedo(self.latitude)
            self.albedo_function = albedo
        else:
            if np.size(albedo) < self.ny and np.size(albedo) == 1:
                self.albedo = np.repeat(albedo, self.ny)
            else:
                self.albedo = albedo
            self.albedo_function = None
        self.F_stellar_constant = F_stellar_constant
        # net total down sw_flux with no atmosphere
        self.solar_latitude_factor = GreyGas.latitudinal_solar_distribution(self.latitude)
        self.F_sw0 = (1 - self.albedo) * self.solar_latitude_factor * self.F_stellar_constant / 4
        self.T0 = self.get_isothermal_temp()
        self.tau_lw_func = tau_lw_func
        self.tau_lw_func_args = tuple(tau_lw_func_args)
        self.tau_sw_func = tau_sw_func
        self.tau_sw_func_args = tau_sw_func_args
        self.sw_tau_is_zero = self.tau_sw_func is None or self.tau_sw_func_args.count(0) > 0
        self.p_interface, self.tau_interface = self.get_p_and_tau_interface_grids()
        self.T = np.ones((self.nz - 1, self.ny)) * self.T0
        self.p = np.zeros((self.nz - 1, self.ny))  # at same height as temperature
        for i in range(self.nz - 1):
            self.p[i, :] = np.mean(self.p_interface[i:i + 2, :], 0)
        self.q, self.tau, _, _ = self.tau_lw_func(self.p, *self.tau_lw_func_args)
        if not self.sw_tau_is_zero:
            self.tau_sw_interface = self.tau_sw_func(self.p_interface, *self.tau_sw_func_args)[1]
            self.q_sw, self.tau_sw, _, _ = self.tau_sw_func(self.p, *self.tau_sw_func_args)
        self.dtau = np.abs(self.tau_interface[1:, :] - self.tau_interface[:-1, :])
        '''radiation fluxes start off at isothermal values'''
        # Initial condition such that energy balance with short wave radiation.
        self.up_lw_flux = np.ones((self.tau_interface.shape[0], self.tau_interface.shape[1])) * self.F_sw0
        # down_lw_flux = 0 at top of atmosphere.
        self.down_lw_flux = np.zeros((self.tau_interface.shape[0], self.tau_interface.shape[1]))
        self.up_sw_flux, self.down_sw_flux = self.get_sw_flux(isothermal=True)
        self.net_flux = self.up_lw_flux - self.down_lw_flux + self.up_sw_flux - self.down_sw_flux
        self.time_step_info = None

    def get_p_and_tau_interface_grids(self, nz_multiplier_param=100000, q_thresh_info_percentile=75,
                                      q_thresh_info_max=1000, log_p_min_sep=0.1, tau_min_sep=1e-3):
        """
        Get pressure and optical depth grids together so have good separation in both.

        :param nz_multiplier_param: float, optional.
            Each local maxima in mass concentration, q_max, will have nz_multiplier_param*q_max grid points
            associated with it if nz is 'auto'.
            default: 100000
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

        '''Get mass concentrations so ensure have grid points around local maxima'''
        q = self.tau_lw_func(p_interface, *self.tau_lw_func_args)[0]
        small = 1e-10
        if not self.sw_tau_is_zero:
            q_sw = self.tau_sw_func(p_interface, *self.tau_sw_func_args)[0]
            q_sw_local_maxima_index = argrelextrema(np.insert(q_sw, 0, q_sw[1] - small), np.greater)[0] - 1
            q_sw_local_maxima_index = q_sw_local_maxima_index[q_sw_local_maxima_index >= 0]
            q = q + q_sw
        cum_q = np.cumsum(q)
        # to ensure find local maxima if at surface, add value one index away from surface below surface too.
        # subtract small amount to correct for case where value at surface = value one away from surface
        q_local_maxima_index = argrelextrema(np.insert(q, 0, q[1] - small), np.greater)[0] - 1
        q_local_maxima_index = q_local_maxima_index[q_local_maxima_index >= 0]
        if not self.sw_tau_is_zero:
            q_local_maxima_index = np.sort(np.concatenate((q_local_maxima_index, q_sw_local_maxima_index)))
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
        tau_interface = self.tau_lw_func(np.tile(p_interface, (self.ny, 1)).transpose(), *self.tau_lw_func_args)[1]
        delta_tau = abs(np.ediff1d(tau_interface))
        to_correct = np.where(delta_log_p > log_p_min_sep)[0]
        to_correct = to_correct[np.where(delta_tau[to_correct] > tau_min_sep)]  # need significant concentration
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
        tau_interface = self.tau_lw_func(p_interface, *self.tau_lw_func_args)[1]
        return p_interface, tau_interface

    @staticmethod
    def latitudinal_solar_distribution(latitude, c=0.477):
        """
        Returns annually averaged solar radiation as function of latitude factor using equation 13.2.6 and
        normalised according to 13.2.4 in Atmospheric circulation dynamics and General Circulation Models.

        :param latitude: list.
            list of latitudes to find solar flux at.
        :param c: float.
            empirical value, default is from North, 1975.
        """
        if np.size(latitude) > 1:
            delta_latitude = (np.radians(latitude[1]) - np.radians(latitude[0]))
            lat_dist = 1 - 0.5 * c * (3 * np.sin(np.radians(latitude)) ** 2 - 1)
            # normalise so integral of 0.5 * lat_dist x cos(latitude) = 1.
            norm_factor = np.trapz(0.5 * lat_dist * np.cos(np.radians(latitude)), np.radians(latitude))
            lat_dist = lat_dist / norm_factor  # ensure normalised so sum is 1.
        else:
            lat_dist = 1
        return lat_dist


    def get_isothermal_temp(self):
        """Get initial isothermal temperature to satisfy radiation balance in absence of atmosphere"""
        return np.power(self.F_sw0 / sigma, 1 / 4)

    def get_lw_flux(self):
        """
        Determines up_lw_flux from boundary condition that at top of atmosphere this balances net sw_flux.
        Determines down_lw_flux from boundary condition that at top of atmosphere this equals zero.
        """
        '''Explicit method below gives oscillations and weird start
        self.up_lw_flux[:-1, :] = (
                    self.up_lw_flux[1:, :] * np.exp(dtau) +
                    sigma * np.power(self.T, 4) * (1 - np.exp(dtau)))
        self.down_lw_flux[:-1, :] = (
                self.down_lw_flux[1:, :] * np.exp(-dtau) +
                sigma * np.power(self.T, 4) * (1 - np.exp(-dtau)))'''
        # use for loop because use updated values in next value of i so converge quicker.
        # set top of atmosphere level everytime to take account of any changing albedo or stellar_constant
        self.up_lw_flux[-1, :] = (1 - self.albedo) * self.solar_latitude_factor * self.F_stellar_constant / 4
        for i in range(self.T.shape[0] - 1, -1, -1):
            # think up_flux routine is a bit questionable, heating is from below so from physical point of view,
            # should compute i+1 flux from i flux with i+1/2 temperature.
            # but then can't use upper atmosphere boundary condition.
            self.up_lw_flux[i, :] = (
                    self.up_lw_flux[i + 1, :] * np.exp(self.dtau[i]) +
                    sigma * np.power(self.T[i, :], 4) * (1 - np.exp(self.dtau[i])))
            self.down_lw_flux[i, :] = (
                    self.down_lw_flux[i + 1, :] * np.exp(-self.dtau[i]) +
                    sigma * np.power(self.T[i, :], 4) * (1 - np.exp(-self.dtau[i])))

    def get_sw_flux(self, isothermal=False):
        """
        Determines short wave fluxes assuming that at these frequencies and temperatures,
        emission of atmosphere itself is zero. I.e. no overlap between stellar and planetary spectrum.
        BCs are that up_sw_flux equals reflected sw_flux at top of atmosphere. down_sw_flux equals incoming
        stellar sw_flux at top of atmosphere.

        :param isothermal: boolean, optional.
            If Isothermal, fluxes are those in absence of atmosphere.
        """
        up_sw_flux = np.ones((self.tau_interface.shape[0], self.tau_interface.shape[1])) * \
                     self.albedo * self.solar_latitude_factor * self.F_stellar_constant / 4
        down_sw_flux = np.ones((self.tau_interface.shape[0], self.tau_interface.shape[1])) * \
                       self.solar_latitude_factor * self.F_stellar_constant / 4
        if not self.sw_tau_is_zero and isothermal is False:
            up_sw_flux = up_sw_flux * np.exp(self.tau_sw_interface)
            down_sw_flux = down_sw_flux * np.exp(-self.tau_sw_interface)
        return up_sw_flux, down_sw_flux

    def update_temp(self, t, T_initial=None, changing_tau=False, convective_adjust=False,
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
        if changing_tau:
            self.update_grid()
        if t == 0 and T_initial is not None:
            self.T = T_initial
        # Use radiation fluxes to update temperature
        self.get_lw_flux()
        self.up_sw_flux, self.down_sw_flux = self.get_sw_flux()
        # net_lw_flux should be zero everywhere in equilibrium
        net_flux = self.up_lw_flux - self.down_lw_flux + self.up_sw_flux - self.down_sw_flux
        # Use finite volume method
        T_tendency = g / c_p_dry * (
                net_flux[1:, :] - net_flux[:-1, :]) / (
                             self.p_interface[1:, :] - self.p_interface[:-1, :])

        if t > 0 and changing_tau is False:
            # Remove grid levels which are stagnating to reach equilibrium more quickly.
            levels_to_update = np.where(abs(net_flux[:-1].flatten()) > net_flux_thresh)[0]
            levels_to_update = np.setdiff1d(levels_to_update, self.time_step_info['RemoveInd'])
        else:
            # always update every level.
            levels_to_update = np.arange((self.nz - 1) * self.ny)
            delta_net_flux = 1e6  # any large number
        if changing_tau is False:
            delta_net_flux = np.percentile(abs(net_flux - self.net_flux), net_flux_percentile)
        if len(levels_to_update) > 0:
            self.update_time_step(T_tendency, levels_to_update)
        self.net_flux = net_flux
        update_ind = np.unravel_index(levels_to_update, self.T.shape)  # from 1d to 2d
        self.T[update_ind] = self.T[update_ind] + self.time_step_info['dt'] * T_tendency[update_ind]
        if convective_adjust:
            self.T = convective_adjustment(self.p[:, 0], self.T)
        t = t + self.time_step_info['dt']
        return t, delta_net_flux

    def update_time_step(self, T_tendency, levels_to_update, temp_change=1, delta_temp_change=0.001):
        """
        This finds the time step such that precisely one pressure level will have its temperature changed by
        self.time_step_info['DeltaT'] kelvin. This defaults to 1 kelvin and will be lowered if not converging.
        Info used to compute time step is stored in self.time_step_info.

        :param T_tendency: numpy array.
            Rate at which temperature is changing at each level. Units = K/s
        :param levels_to_update: list.
            Temperature is only being updated at grid levels indicated by this.
        :param temp_change: float, optional.
            Time step is found so that at least one level will have a temperature change equal to this.
            default: 1K.
        :param delta_temp_change: float, optional.
            If not converging, temp_change will be lowered by delta_temp_change.
            default: 0.001K.
        """
        if self.time_step_info is None:
            # have one temperature change of DeltaT K at each step.
            # nSameMaxInd and nSameMaxInd2 are used to find levels that are stagnating or oscillating.
            # These are then added to RemoveInd so will no longer update.
            self.time_step_info = {'DeltaT': temp_change, 'MaxTend': 0, 'MaxTendInd': -1,
                                   'DeltaT_step': delta_temp_change, 'dt': 0,
                                   'MaxDeltaT': temp_change, 'nSameMaxInd': 0, 'nSameMaxInd2': 0,
                                   'RemoveInd': []}

        """Find time step based on level that has temperature that is changing the most"""
        MaxTendInd = levels_to_update[np.argmax(abs(T_tendency.flatten()[levels_to_update]))]
        MaxTend = T_tendency.flatten()[MaxTendInd]
        if (MaxTendInd == self.time_step_info['MaxTendInd'] and
                np.sign(MaxTend) != np.sign(self.time_step_info['MaxTend'])):
            # if get oscillations in temperature, lower time step.
            self.time_step_info['DeltaT'] = self.time_step_info['DeltaT'] - self.time_step_info['DeltaT_step']
            self.time_step_info['nSameMaxInd'] += 1
            if self.time_step_info['DeltaT'] < self.time_step_info['DeltaT_step']:
                self.time_step_info['DeltaT'] = self.time_step_info['DeltaT_step']
        elif (MaxTendInd == self.time_step_info['MaxTendInd'] and
              np.sign(MaxTend) == np.sign(self.time_step_info['MaxTend']) and
              self.time_step_info['DeltaT'] < self.time_step_info['MaxDeltaT']):
            # if no oscillations anymore, start increasing time step.
            self.time_step_info['DeltaT'] = self.time_step_info['DeltaT'] + self.time_step_info['DeltaT_step']
            self.time_step_info['nSameMaxInd2'] += 1
            if self.time_step_info['DeltaT'] > self.time_step_info['MaxDeltaT']:
                self.time_step_info['DeltaT'] = self.time_step_info['MaxDeltaT']
        else:
            self.time_step_info['nSameMaxInd'] = 0
            self.time_step_info['nSameMaxInd2'] = 0

        if (self.time_step_info['nSameMaxInd'] > 0 and self.time_step_info['nSameMaxInd2'] > 10) or (
                self.time_step_info['nSameMaxInd'] > 100 and self.time_step_info['nSameMaxInd2'] == 0) or (
                len(self.time_step_info['RemoveInd']) > 5 and self.time_step_info['nSameMaxInd'] +
                self.time_step_info['nSameMaxInd2'] > 0):
            # If one level stagnates or oscillates for prolonged period, then stop updating this level.
            self.time_step_info['RemoveInd'].append(MaxTendInd)
            self.time_step_info['nSameMaxInd'] = 0
            self.time_step_info['nSameMaxInd2'] = 0

        self.time_step_info['MaxTendInd'] = MaxTendInd
        self.time_step_info['MaxTend'] = MaxTend
        self.time_step_info['dt'] = float(self.time_step_info['DeltaT'] / abs(MaxTend).max())
        if np.isinf(self.time_step_info['dt']):
            # Set to 1 day time step if calculated it as infinite.
            self.time_step_info['dt'] = 24 * 60 ** 2

    def update_grid(self):
        """
        This updates the optical depth and mass concentration grid values for long and short wave if
        changing_tau is True in update_temp.
        """
        old_sw_tau_is_zero = self.sw_tau_is_zero
        self.sw_tau_is_zero = self.tau_sw_func is None or self.tau_sw_func_args.count(0) > 0
        self.tau_interface = self.tau_lw_func(self.p_interface, *self.tau_lw_func_args)[1]
        self.q, self.tau, _, _ = self.tau_lw_func(self.p, *self.tau_lw_func_args)
        self.dtau = np.abs(self.tau_interface[1:, :] - self.tau_interface[:-1, :])
        if (not self.sw_tau_is_zero) or (not old_sw_tau_is_zero):
            self.tau_sw_interface = self.tau_sw_func(self.p_interface, *self.tau_sw_func_args)[1]
            self.q_sw, self.tau_sw, _, _ = self.tau_sw_func(self.p, *self.tau_sw_func_args)

    def check_equilibrium(self, delta_net_flux, flux_thresh=1e-3):
        """
        This checks if either net flux has reached zero or it is not changing anymore. In which case
        it will return True indicating equilibrium has been achieved.

        :param delta_net_flux: float.
            change in flux between time steps
        :param flux_thresh: float, optional.
            Threshold to achieve equilibrium
            default: 1e-3
        :return:
        """
        if max(abs(self.net_flux.flatten())) < flux_thresh or delta_net_flux < flux_thresh:
            equilibrium = True
        else:
            equilibrium = False
        return equilibrium

    def save_data(self, data_dict, t):
        """
        This appends the time and current Temperature to the data_dict. It will also add optical depth and
        flux data if they are already in data_dict.

        :param data_dict: dictionary.
            Must contain 't' and 'T' keys. Can also contain 'tau' and 'flux' keys.
            The info in this dictionary is used to pass to plot_animate.
        :param t: float.
            Current time.
        :return:
            data_dict
        """
        data_dict['t'].append(t)
        data_dict['T'].append(self.T.copy())
        if "tau" in data_dict:
            data_dict['tau']['lw'].append(self.tau.copy())
            data_dict['tau']['sw'].append(self.tau_sw.copy())
        if "flux" in data_dict:
            data_dict['flux']['lw_up'].append(self.up_lw_flux.copy())
            data_dict['flux']['lw_down'].append(self.down_lw_flux.copy())
            data_dict['flux']['sw_up'].append(self.up_sw_flux.copy())
            data_dict['flux']['sw_down'].append(self.down_sw_flux.copy())
        return data_dict

    def evolve_to_equilibrium(self, data_dict=None, flux_thresh=1e-3, T_initial=None, convective_adjust=False):
        """
        This updates the temperature profile until the equilibrium condition is reached.

        :param data_dict: dictionary, optional.
            Must contain 't' and 'T' keys. Can also contain 'tau' and 'flux' keys.
            The info in this dictionary is used to pass to plot_animate.
            Must contain at least one value in each array.
            If not provided, will set first value to be t=0 and T = T_initial or self.T
            default: None
        :param flux_thresh: float, optional.
            Threshold in delta_net_flux to achieve equilibrium
            default: 1e-3
        :param T_initial: numpy array, optional.
            Temperature profile at t=0. If not provided, will use self.T
            default: None
        :param convective_adjust: boolean, optional.
            Whether the temperature profile should adjust to stay stable with respect to convection.
            default: False
        :return:
            data_dict
        """
        if data_dict is None:
            if T_initial is None:
                T_initial = self.T.copy()
            data_dict = {'t': [0], 'T': [T_initial.copy()]}
        t = data_dict['t'][-1]
        equilibrium = False
        while not equilibrium:
            t, delta_net_flux = self.update_temp(t, T_initial, changing_tau=False,
                                                 convective_adjust=convective_adjust)
            data_dict = self.save_data(data_dict, t)
            equilibrium = self.check_equilibrium(delta_net_flux, flux_thresh)
            if min(self.T.flatten()) < 0:
                raise ValueError('Temperature is below zero')
        # set RemoveInd empty so will evolve all pressure levels if continue after this
        self.time_step_info['RemoveInd'] = []
        return data_dict

    def equilibrium_sol(self, convective_adjust=False):
        """
        Calculates the analytic equilibrium solution given the current optical depth grids.
        If short wave optical depth is non zero, an analytic solution can only be computed if both functions are
        exponential and the ratio in the power argument is an integer and less than 10.
        If cannot compute an analytic equilibrium solution, the equilibrium solution in the absence of any
        short wave optical depth is returned.
        correct_solution will be False if the short wave optical depth had to be set to zero.

        :param convective_adjust: boolean, optional.
            Whether the temperature profile should adjust to stay stable with respect to convection.
            default: False

        :return:
        up_lw_flux_eqb
        down_lw_flux_eqb
        T_eqb
        up_sw_flux_eqb
        down_sw_flux_eqb
        correct_solution
        """
        '''Check if an analytic solution can be found'''
        if self.sw_tau_is_zero:
            correct_solution = True
        elif self.tau_lw_func.__name__ == 'exponential' and self.tau_sw_func.__name__ == 'exponential':
            alpha_lw = od.get_exponential_alpha(self.tau_lw_func_args[0])
            alpha_sw = od.get_exponential_alpha(self.tau_sw_func_args[0])
            power_ratio = alpha_lw / alpha_sw
            if abs(round(power_ratio) - power_ratio) < 1e-5 and power_ratio < 10:
                correct_solution = True
            else:
                warnings.warn("\nCan only compute exact solution if the ratio of long wave alpha parameter, " +
                              str.format('{0:1.1e}', alpha_lw) + ", to short wave alpha parameter, " +
                              str.format('{0:1.1e}', alpha_sw) + ", is an integer and <10."
                                                                 "\nCurrently ratio is " + str(power_ratio) +
                              "\nReturned equilibrium solution is that with short wave optical depth = 0 everywhere.")
                correct_solution = False
        else:
            lw_name = self.tau_lw_func.__name__
            sw_name = self.tau_sw_func.__name__
            warnings.warn("\nCan only compute exact solution if both long wave and short wave optical depth functions"
                          " are exponential.\nSelected functions are:\nLong wave - " + lw_name + "\nShort Wave - "
                          + sw_name + "\nReturned equilibrium solution is that with short wave optical depth = 0 everywhere.")
            correct_solution = False

        if not self.sw_tau_is_zero and correct_solution:
            '''Find analytic solution with short wave tau'''
            swEqb = ShortWavelengthEqbCalc(self.F_stellar_constant, self.albedo,
                                           self.tau_lw_func_args, self.tau_sw_func_args,
                                           self.tau_lw_func, self.tau_sw_func)
            up_lw_flux_eqb = swEqb.up_lw_flux(self.tau_sw_interface)
            down_lw_flux_eqb = swEqb.down_lw_flux(self.tau_sw_interface)
            T_eqb = swEqb.T(self.tau_sw)  # Temperature at cell centre
            up_sw_flux_eqb = swEqb.up_sw_flux(self.tau_sw_interface)
            down_sw_flux_eqb = swEqb.down_sw_flux(self.tau_sw_interface)
        else:
            '''Find analytic solution with no short wave tau'''
            # Fluxes at interface
            up_lw_flux_eqb = 0.5 * self.F_sw0 * (2 + self.tau_interface)
            down_lw_flux_eqb = 0.5 * self.F_sw0 * self.tau_interface
            # Temperature at centre
            T_eqb = np.power((self.F_sw0 / (2 * sigma)) * (1 + self.tau), 1 / 4)
            up_sw_flux_eqb = np.ones(np.shape(up_lw_flux_eqb)) * self.albedo * self.F_stellar_constant / 4
            down_sw_flux_eqb = np.ones(np.shape(up_lw_flux_eqb)) * self.F_stellar_constant / 4
        if convective_adjust:
            T_eqb = convective_adjustment(self.p[:, 0], T_eqb)
        return up_lw_flux_eqb, down_lw_flux_eqb, T_eqb, up_sw_flux_eqb, down_sw_flux_eqb, correct_solution

    def plot_eqb(self, up_lw_flux_eqb, down_lw_flux_eqb, T_eqb, up_sw_flux_eqb, down_sw_flux_eqb):
        """
        This plots optical depth profiles - equilibrium temperature profiles - equilibrium flux profiles.
        All params returned by equilibrium_sol function.

        :param up_lw_flux_eqb: equilibrium upward long wave flux
        :param down_lw_flux_eqb: equilibrium downward long wave flux
        :param T_eqb: equilibrium temperature
        :param up_sw_flux_eqb: equilibrium upward short wave flux
        :param down_sw_flux_eqb: equilibrium downward short wave flux
        """
        fig, ax = plt.subplots(1, 3, sharey=True, figsize=(12, 5))
        sw_color = '#1f77b4'
        lw_color = '#ff7f0e'
        if not self.sw_tau_is_zero:
            ax[0].plot(self.tau_sw_interface, self.p_interface, label=r'short wave, $\tau_{sw}$', color=sw_color)
        ax[0].plot(self.tau_interface, self.p_interface, label=r'long wave, $\tau_{lw}$', color=lw_color)
        ax[0].set_xlabel(r'Optical depth, $\tau$')
        ax[0].set_ylabel('Pressure / Pa')
        ax[1].plot(T_eqb, self.p, label=r'$\tau_{sw}\neq0$', color=sw_color)
        ax[1].set_xlabel('Temperature / K')
        net_flux = up_lw_flux_eqb + up_sw_flux_eqb - down_lw_flux_eqb - down_sw_flux_eqb
        F_norm = self.F_stellar_constant / 4
        ax[2].plot(up_sw_flux_eqb / F_norm, self.p_interface, color=sw_color)
        if not self.sw_tau_is_zero:
            ax[2].plot(-down_sw_flux_eqb / F_norm, self.p_interface, color=sw_color, label=r'$F_{sw}(\tau_{sw}\neq0)$')
            ax[2].plot(up_lw_flux_eqb / F_norm, self.p_interface, color=lw_color, label=r'$F_{lw}(\tau_{sw}\neq0)$')
        else:
            ax[2].plot(-down_sw_flux_eqb / F_norm, self.p_interface, color=sw_color, label=r'$F_{sw}$')
            ax[2].plot(up_lw_flux_eqb / F_norm, self.p_interface, color=lw_color, label=r'$F_{lw}$')
        ax[2].plot(-down_lw_flux_eqb / F_norm, self.p_interface, color=lw_color)
        ax[2].plot(net_flux / F_norm, self.p_interface, label=r'$F_{net}$', color='#d62728')
        ax[2].set_xlabel(r'Radiation Flux, $F$, as fraction of Incoming Solar, $\frac{F^\odot}{4}$')
        ax[0].invert_yaxis()
        if not self.sw_tau_is_zero:
            ax[0].plot(self.tau_sw_interface * 0, self.p_interface, color=sw_color,
                       linestyle='dotted', label=r'$\tau_{sw}=0$')
            ax[0].legend()
            no_sw_world = GreyGas(self.nz, self.ny, self.tau_lw_func, self.tau_lw_func_args)
            up_lw_no_sw, down_lw_no_sw, T_eqb_no_sw, up_sw_no_sw, down_sw_no_sw, _ = no_sw_world.equilibrium_sol()
            ax[1].plot(T_eqb_no_sw, no_sw_world.p, label=r'$\tau_{sw}=0$', color=sw_color, linestyle='dotted')
            ax[1].legend()
            ax[2].plot(up_sw_no_sw / F_norm, no_sw_world.p_interface, color=sw_color, linestyle='dotted',
                       label=r'$F_{sw}(\tau_{sw}=0)$')
            ax[2].plot(-down_sw_no_sw / F_norm, no_sw_world.p_interface, color=sw_color, linestyle='dotted')
            ax[2].plot(up_lw_no_sw / F_norm, no_sw_world.p_interface, color=lw_color, linestyle='dotted',
                       label=r'$F_{lw}(\tau_{sw}=0)$')
            ax[2].plot(-down_lw_no_sw / F_norm, no_sw_world.p_interface, color=lw_color, linestyle='dotted')
        ax[2].legend()

    def plot_animate(self, T_array, t_array, T_eqb=None, correct_solution=True, tau_array=None, flux_array=None,
                     log_axis=True, nPlotFrames=100, fract_frames_at_start=0.25, start_step=3, show_last_frame=False):
        """
        This plots an animation showing the evolution of the temperature profile and optical depth profile with time.

        :param T_array: list of numpy arrays.
            T_array[i] is the temperature profile at time t_array[i]
        :param t_array: list.
            times where temperature profile was updated in the simulation.
        :param T_eqb: numpy array, optional.
            If given, analytic equilibrium solution will also be plotted.
            default: None.
        :param correct_solution: boolean, optional.
            If T_eqb given, this indicates whether the analytic solution was correct. Affects label.
        :param tau_array: dictionary, optional.
            tau_array['lw'][i] is long wave optical depth profile at time t_array[i].
            tau_array['sw'][i] is short wave optical depth profile at time t_array[i].
            If given, will add second subplot showing evolution of long wave optical depth.
            Only really makes sense to plot if optical depth was changing with time.
            default: None.
        :param flux_array: dictionary, optional.
            flux_array['lw_up'][i] is long wave upward flux at time t_array[i].
            flux_array['lw_down'][i] is long wave downward flux at time t_array[i].
            flux_array['sw_up'][i] is short wave upward flux at time t_array[i].
            flux_array['sw_down'][i] is short wave downward flux at time t_array[i].
            If given will add subplot showing evolution of flux with time.
            default: None.
        :param log_axis: boolean, optional.
            Whether to have pressure on the y-axis as a log-scale.
            default: True.
        :param nPlotFrames: integer, optional.
            Number of frames to show in the animation.
            default: 100.
        :param fract_frames_at_start: float, optional.
            fract_frames_at_start*nPlotFrames of the animation frames will be at the start.
            The remainder will be equally spaced amongst the remaining times.
            default: 0.25
        :param start_step: integer, optional.
            The step size of the first fract_frames_at_start*nPlotFrames frames.
            A value of 2 would mean every other time in the first fract_frames_at_start*nPlotFrames*2 frames.
            default: 3
        :param show_last_frame:  boolean, optional.
            If True, will plot the last frame collected. Otherwise will only plot till temperature stops changing.
            default: False
        """
        '''Get subsection of data for plotting'''
        if self.ny > 1 and tau_array is not None:
            for key in [*tau_array]:
                # assume tau is same at all latitudes so choose first latitude.
                tau_array[key] = np.array(tau_array[key])[:, :, 0]
        F_norm = self.F_stellar_constant / 4  # normalisation for flux plots
        if len(T_array) > nPlotFrames:
            start_end_ind = start_step * int(fract_frames_at_start * nPlotFrames)
            use_plot_start = np.arange(0, start_end_ind, start_step)
            # find max_index beyond which temperature is constant
            index_where_temp_diff_small = np.where(np.percentile(np.abs(np.diff(T_array, axis=0)), 99, axis=1) < 0.01)[
                0]
            index_sep = np.where(np.ediff1d(index_where_temp_diff_small) > 1)[0]
            if len(index_sep) == 0:
                if len(index_where_temp_diff_small) == 0:
                    max_index = len(T_array) - 1
                else:
                    max_index = index_where_temp_diff_small[0] + 1
            else:
                max_index = index_where_temp_diff_small[max(index_sep) + 1] + 1
            if show_last_frame:
                max_index = len(T_array) - 1
            use_plot_end = np.linspace(start_end_ind, max_index,
                                       int((1 - fract_frames_at_start) * nPlotFrames), dtype=int)
            use_plot = np.unique(np.concatenate((use_plot_start, use_plot_end)))
            T_plot = np.array(T_array)[use_plot]
            t_plot = np.array(t_array)[use_plot]
            if tau_array is not None:
                tau_lw_plot = np.array(tau_array['lw'])[use_plot]
                tau_sw_plot = np.array(tau_array['sw'])[use_plot]
            if flux_array is not None and self.ny == 1:
                lw_up_flux_plot = np.array(flux_array['lw_up'])[use_plot] / F_norm
                lw_down_flux_plot = np.array(flux_array['lw_down'])[use_plot] / F_norm
                sw_up_flux_plot = np.array(flux_array['sw_up'])[use_plot] / F_norm
                sw_down_flux_plot = np.array(flux_array['sw_down'])[use_plot] / F_norm
        else:
            T_plot = np.array(T_array)
            t_plot = np.array(t_array)
            if tau_array is not None:
                tau_lw_plot = np.array(tau_array['lw'])
                tau_sw_plot = np.array(tau_array['sw'])
            if flux_array is not None and self.ny == 1:
                lw_up_flux_plot = np.array(flux_array['lw_up']) / F_norm
                lw_down_flux_plot = np.array(flux_array['lw_down']) / F_norm
                sw_up_flux_plot = np.array(flux_array['sw_up']) / F_norm
                sw_down_flux_plot = np.array(flux_array['sw_down']) / F_norm
        if flux_array is not None and self.ny == 1:
            net_flux_plot = lw_up_flux_plot + sw_up_flux_plot - lw_down_flux_plot - sw_down_flux_plot

        '''Set up basic plot info'''
        if self.ny > 1:
            if tau_array is not None:
                fig, axs = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
                gs = axs[1, 1].get_gridspec()
                for ax in axs[-1, :]:
                    ax.remove()
                axTemp = fig.add_subplot(gs[-1, :])
                axs[0, 1].get_shared_y_axes().join(axs[0, 1], axs[0, 0])
                axs[0, 1].get_shared_x_axes().join(axs[0, 1], axTemp)
                axColor = axs[0, 1]
                ax_tau = axs[0, 0]
            else:
                fig, (axColor, axTemp) = \
                    plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]})
        else:
            nPlots = 1 + int(tau_array is not None) + int(flux_array is not None)
            if nPlots > 1:
                fig, axs = plt.subplots(1, nPlots, sharey=True, figsize=(6 * nPlots, 5))
                ax = axs[0]
            else:
                fig, ax = plt.subplots(1, 1)

        if tau_array is not None:
            tau_min = min(min(i for v in tau_array[key] for i in v) for key in [*tau_array]) - 1
            tau_max = max(max(i for v in tau_array[key] for i in v) for key in [*tau_array]) + 1
        if flux_array is not None and self.ny == 1:
            flux_min = -max(max(i for v in flux_array[key] for i in v) for key in ['lw_down', 'sw_down']) / F_norm - 0.1
            flux_max = max(max(i for v in flux_array[key] for i in v) for key in ['lw_up', 'sw_up']) / F_norm + 0.1

        T_min = min([min(T_plot[i].flatten()) for i in range(len(T_plot))]) - 10
        T_max = max([max(T_plot[i].flatten()) for i in range(len(T_plot))]) + 10
        if T_eqb is not None:
            T_min = min([min(T_eqb.flatten()) - 10, T_min])
            T_max = max([max(T_eqb.flatten()) + 10, T_max])

        lw_color = '#ff7f0e'
        sw_color = '#1f77b4'

        def animate(i, grey_world):
            '''What to do at each frame of animation'''
            ax.clear()
            ax.plot(np.ones((grey_world.nz - 1, grey_world.ny)) * grey_world.T0, grey_world.p,
                    label='Isothermal', color=sw_color)
            if T_eqb is not None:
                if correct_solution and not grey_world.sw_tau_is_zero:
                    Teqb_label = r'Radiative Equilibrium, $\tau_{sw}\neq0$'
                    Tcurrent_label = r'Current, $\tau_{sw}\neq0$'
                elif correct_solution:
                    Teqb_label = r'Radiative Equilibrium, $\tau_{sw}=0$'
                    Tcurrent_label = r'Current, $\tau_{sw}=0$'
                else:
                    Teqb_label = r'Radiative Equilibrium, $\tau_{sw}=0$ (Wrong)'
                    Tcurrent_label = r'Current, $\tau_{sw}\neq0$'
                ax.plot(T_eqb, grey_world.p, label=Teqb_label, color=lw_color)
            else:
                Tcurrent_label = 'Current'
            if tau_array is not None:
                Tcurrent_label = 'Current'
            ax.plot(T_plot[i], grey_world.p, label=Tcurrent_label, color='#d62728')
            if log_axis:
                ax.set_yscale('log')
            ax.axes.invert_yaxis()
            ax.set_xlabel('Temperature / K')
            ax.set_ylabel('Pressure / Pa')
            ax.set_xlim((T_min, T_max))
            ax.legend()
            if tau_array is not None:
                axs[1].clear()
                axs[1].plot(tau_sw_plot[i], grey_world.p, label='short wave', color=sw_color)
                axs[1].plot(tau_lw_plot[i], grey_world.p, label='long wave', color=lw_color)
                axs[1].set_xlabel(r'$\tau$')
                axs[1].set_xlim((tau_min, tau_max))
                if log_axis:
                    axs[1].set_yscale('log')
                axs[1].legend()
            if flux_array is not None:
                axs[-1].clear()
                axs[-1].plot(sw_up_flux_plot[0], grey_world.p_interface, color=sw_color,
                             linestyle='dotted', label=r'$F_{sw}(t=0)$')
                axs[-1].plot(-sw_down_flux_plot[0], grey_world.p_interface, color=sw_color,
                             linestyle='dotted')
                axs[-1].plot(lw_up_flux_plot[0], grey_world.p_interface, color=lw_color,
                             linestyle='dotted', label=r'$F_{lw}(t=0)$')
                axs[-1].plot(-lw_down_flux_plot[0], grey_world.p_interface, color=lw_color,
                             linestyle='dotted')
                axs[-1].plot(sw_up_flux_plot[i], grey_world.p_interface, color=sw_color,
                             label=r'$F_{sw}$')
                axs[-1].plot(-sw_down_flux_plot[i], grey_world.p_interface, color=sw_color)
                axs[-1].plot(lw_up_flux_plot[i], grey_world.p_interface, color=lw_color,
                             label=r'$F_{lw}$')
                axs[-1].plot(-lw_down_flux_plot[i], grey_world.p_interface, color=lw_color)
                axs[-1].plot(net_flux_plot[i], self.p_interface, label=r'$F_{net}$', color='#d62728')
                axs[-1].set_xlabel(r'Radiation Flux, $F$, as fraction of Incoming Solar, $\frac{F^\odot}{4}$')
                flux_max_i = max([max(sw_up_flux_plot[i]), max(lw_up_flux_plot[i])])
                flux_min_i = -max([max(sw_down_flux_plot[i]), max(lw_down_flux_plot[i])])
                if flux_max > 5 and flux_max_i < 5:
                    current_flux_max = 5
                else:
                    current_flux_max = flux_max
                if flux_min < -5 and flux_min_i > -5:
                    current_flux_min = -5
                else:
                    current_flux_min = flux_min
                axs[-1].set_xlim((current_flux_min, current_flux_max))
                if log_axis:
                    axs[-1].set_yscale('log')
                axs[-1].legend()
            t_full_days = t_plot[i] / (24 * 60 ** 2)
            t_years, t_days = divmod(t_full_days, 365)
            Title = "{:.0f}".format(t_years) + " Years and " + "{:.1f}".format(t_days) + " Days"
            ax.text(0.5, 1.01, Title, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)

        if self.ny > 1:
            div = make_axes_locatable(axColor)
            cax = div.append_axes('right', '5%', '5%')
            plot_extent = [self.latitude.min(), self.latitude.max(),
                           self.p[:, 0].min(), self.p[:, 0].max()]
            n_labels_keep = min([len(self.latitude), 6])
            keep_x_ticks = np.linspace(0, len(self.latitude) - 1, n_labels_keep, dtype='int')
            miss_x_ticks = np.setdiff1d(np.linspace(0, len(self.latitude) - 1, len(self.latitude), dtype='int'),
                                        keep_x_ticks)
            X, Y = np.meshgrid(np.array(self.latitude), self.p[:, 0])

        def animate2D(i, grey_world):
            cax.cla()
            axColor.clear()
            axTemp.clear()
            im = axColor.pcolormesh(X, Y, T_plot[i], cmap='bwr')
            im.set_clim((T_min, T_max))
            axColor.axes.invert_yaxis()
            if log_axis:
                axColor.set_yscale('log')
            axColor.set_xticks(grey_world.latitude)

            axTemp.plot(grey_world.latitude, T_plot[i][0], label='current')
            axTemp.plot(grey_world.latitude, T_plot[0][0], label='initial')
            axTemp.set_ylim((T_min, T_max))
            axTemp.set_xlabel('Latitude')
            axTemp.set_ylabel('Surface Temperature / K')
            axTemp.legend(loc='upper right')
            t_full_days = t_plot[i] / (24 * 60 ** 2)
            t_years, t_days = divmod(t_full_days, 365)
            Title = "{:.0f}".format(t_years) + " Years and " + "{:.1f}".format(t_days) + " Days"
            axColor.text(0.5, 1.01, Title, horizontalalignment='center', verticalalignment='bottom',
                         transform=axColor.transAxes)
            cb = fig.colorbar(im, cax=cax)
            cb.set_label('Temperature / K')
            if tau_array is not None:
                ax_tau.clear()
                ax_tau.plot(tau_sw_plot[i], grey_world.p[:, 0], label='short wave', color=sw_color)
                ax_tau.plot(tau_lw_plot[i], grey_world.p[:, 0], label='long wave', color=lw_color)
                ax_tau.set_xlabel(r'$\tau$')
                ax_tau.set_xlim((tau_min, tau_max))
                if log_axis:
                    ax_tau.set_yscale('log')
                ax_tau.axes.invert_yaxis()
                ax_tau.legend(loc='upper right')
                ax_tau.set_ylabel('Pressure / Pa')
                axTemp.set_xticks(grey_world.latitude)
                [l.set_visible(False) for (j, l) in enumerate(axColor.xaxis.get_ticklabels())]
            else:
                axColor.set_ylabel('Pressure / Pa')
                axColor.set_xticks(grey_world.latitude)
            [l.set_visible(False) for (j, l) in enumerate(axTemp.xaxis.get_ticklabels()) if j in miss_x_ticks]

        if self.ny == 1:
            anim = FuncAnimation(fig, animate,
                                 frames=np.size(t_plot), interval=100, blit=False, repeat_delay=2000, fargs=(self,))
        else:
            anim = FuncAnimation(fig, animate2D,
                                 frames=np.size(t_plot), interval=100, blit=False, repeat_delay=2000, fargs=(self,))

        plot_backend = plt.get_backend()
        # if plot_backend == 'TkAgg' and nPlots == 3:
        #     mng = plt.get_current_fig_manager()
        #     mng.resize(*mng.window.maxsize())
        return anim


class ShortWavelengthEqbCalc:

    def __init__(self, F_stellar_const, albedo, lw_args, sw_args,
                 tau1=od.exponential, tau2=od.exponential):
        """
        This calculates the analytical equilibrium temperature and flux profiles for an atmosphere containing
        both a long wave and a short wave affecting gas.

        :param F_stellar_const: float.
            Flux density at surface of planet. Units = W/m^2;
        :param albedo: float.
            Fraction of incoming short wave radiation reflected back to space.
        :param lw_args: tuple.
            args needed to compute long wave optical depth from the function tau1.
        :param sw_args: tuple.
            args needed to compute short wave optical depth from the function tau2.
        :param tau1: function from grey_optical_depth.py, optional.
            function to determine long wave optical depth. Currently only can work with exponential profile
            default: od.exponential
        :param tau2: function from od, optional.
            function to determine short wave optical depth. Currently only can work with exponential profile
            default: od.exponential

        """
        if np.size(albedo) > 1:
            raise ValueError('Must provide a single latitude bin to get analytical solution')
        _, _, tau_lw, self.lw_params = tau1(1, *lw_args)
        _, _, tau_sw, self.sw_params = tau2(1, *sw_args)
        self.all_params = self.lw_params + self.sw_params
        self.power_params = [False, True, False, True]  # use these to cancel values before integrating
        self.F_stellar_const = F_stellar_const
        self.albedo = albedo
        self.t1 = symbols('tau1')
        self.t2 = symbols('tau2')
        self.tau1_diff_tau2_symbol = self.get_tau1_diff_tau2(tau_lw, tau_sw)[2]
        _, self.integ_sol_onlyt2symbol, self.integ_const, self.tau1_diff_tau2_onlyt2symbol = \
            self.optical_depth_integral()
        self.up_sw_flux, self.down_sw_flux, self.up_lw_flux, self.down_lw_flux, self.T = \
            self.get_physical_values()

    def get_tau1_diff_tau2(self, tau1, tau2):
        """
        Gets tau1 as function of tau2 where both are function of pressure initially.
        Then differentiates tau1 with respect to tau2.
        :param tau1: usually long wave optical depth sympy function
        :param tau2: usually short wave optical depth sympy function
        :return:
        tau1_from_tau2: tau1_from_tau2(tau2, args_tau1, args_tau2) is a function to find tau1 from tau2.
        tau1_diff_tau2: tau1_diff_tau2(tau2, args_tau1, args_tau2) is
         a function differential of tau1 by tau2 at tau2
        """
        '''Give each variable a symbol'''
        # both tau1 and tau2 function of pressure, 1st input to both
        p = symbols(chr(97))
        n_params1 = len(signature(tau1).parameters) - 1
        param_symbols1 = (p,) + tuple(symbols(chr(98 + i)) for i in range(n_params1))
        n_params2 = len(signature(tau2).parameters) - 1
        param_symbols2 = (p,) + tuple(symbols(chr(66 + i)) for i in range(n_params2))  # symbols are capitals for tau2
        '''Find pressure from tau2'''
        p_from_tau2 = solve(self.t2 - tau2(*param_symbols2), p)
        '''Find tau1 from tau2'''
        param_symbols1_from2 = list(param_symbols1)
        param_symbols1_from2[0] = p_from_tau2[0]
        param_symbols1_from2 = tuple(param_symbols1_from2)
        tau1_from_tau2_symbol = simplify(solve(self.t1 - tau1(*param_symbols1_from2), self.t1)[0])
        self.final_params = list((self.t2,) + param_symbols1[1:] + param_symbols2[1:])
        tau1_from_tau2 = lambdify(self.final_params, tau1_from_tau2_symbol, "numpy")
        '''Differentiate tau1 with respect to tau2'''
        tau1_diff_t2symbol = simplify(diff(tau1_from_tau2_symbol, self.t2))
        tau1_diff_tau2 = lambdify(self.final_params, tau1_diff_t2symbol, "numpy")
        return tau1_from_tau2, tau1_diff_tau2, tau1_diff_t2symbol

    def optical_depth_integral(self):
        """
        Performs integration required for optical depth calculation combining optical thickness
        in short and long wave.
        :return:
        integ_sol(tau2) will return the value of the integral as a function of short wave optical depth
        integ_sol_onlyt2symbol is the sympy function for the integral where the only symbol is tau2
        i.e. short wave optical depth. All other parameters have been given their numerical values.
        tau1_diff_tau2_onlyt2symbol is the sympy function for tau1 differentiated with respect to tau2.
        The only symbol is tau2 i.e. short wave optical depth.
        All other parameters have been given their numerical values.
        """
        """Sub in actual power values into tau1_diff_tau2 first as can only perform integral for integer powers"""
        # assume second variable is
        power_variable_values = {self.final_params[i + 1]: self.all_params[i]
                                 for i in np.where(self.power_params)[0]}
        non_power_variable_values = {self.final_params[i + 1]: self.all_params[i]
                                     for i in np.where(np.invert(self.power_params))[0]}
        tau1_diff_tau2_symbol = self.tau1_diff_tau2_symbol.subs(power_variable_values)
        tau1_diff_tau2_symbol = self.set_power_to_int(tau1_diff_tau2_symbol)
        tau1_diff_tau2_onlyt2symbol = tau1_diff_tau2_symbol.subs(non_power_variable_values)
        func_to_int = tau1_diff_tau2_symbol * (exp(-self.t2) - self.albedo * exp(self.t2))
        integ_sol_symbol = integrate(func_to_int, self.t2)
        integ_sol_onlyt2symbol = integ_sol_symbol.subs(non_power_variable_values)
        integ_sol = lambdify(self.t2, integ_sol_onlyt2symbol, "numpy")
        integ_const = 1 - self.albedo - integ_sol(0)  # constant found by satisfying down_lw_flux(0) = 0
        return integ_sol, integ_sol_onlyt2symbol, integ_const, tau1_diff_tau2_onlyt2symbol

    def get_physical_values(self):
        """Once tau1 is found as a function of tau2 and integral is performed, we can get the fluxes
        and temperature as a function of tau2, the short wave optical thickness.
        All returns are functions of short wave optical thickness, tau2"""
        up_sw_flux = self.albedo * self.F_stellar_const / 4 * exp(self.t2)
        down_sw_flux = self.F_stellar_const / 4 * exp(-self.t2)
        sigmaT4 = self.F_stellar_const / 8 * (
                (exp(-self.t2) + self.albedo * exp(self.t2)) / self.tau1_diff_tau2_onlyt2symbol +
                self.integ_sol_onlyt2symbol + self.integ_const)
        down_lw_flux = sigmaT4 - F_sun / 8 * (
                (exp(-self.t2) + self.albedo * exp(self.t2)) / self.tau1_diff_tau2_onlyt2symbol +
                exp(-self.t2) - self.albedo * exp(self.t2))
        up_lw_flux = down_lw_flux + down_sw_flux - up_sw_flux  # ensure net flux is zero
        T = (sigmaT4 / sigma) ** 0.25
        up_sw_flux = lambdify(self.t2, up_sw_flux, "numpy")
        down_sw_flux = lambdify(self.t2, down_sw_flux, "numpy")
        down_lw_flux = lambdify(self.t2, down_lw_flux, "numpy")
        up_lw_flux = lambdify(self.t2, up_lw_flux, "numpy")
        T = lambdify(self.t2, T, "numpy")
        return up_sw_flux, down_sw_flux, up_lw_flux, down_lw_flux, T

    @staticmethod
    def set_power_to_int(expr):
        """
        expression only cancels if powers are integers
        this function makes all powers that are near to an integer part of the int class

        :param expr: sympy expression
        """
        set_int_thresh = 1e-5
        largs = list(expr.args)
        for i in range(len(largs)):
            if str(type(largs[i])) == '<class \'sympy.core.power.Pow\'>':
                if abs(int(round(largs[i].args[1])) - largs[i].args[1]) < set_int_thresh:
                    new_pow = int(round(largs[i].args[1]))
                    sub_exp_args = list(largs[i].args)
                    sub_exp_args[1] = new_pow
                    largs[i] = largs[i].func(*sub_exp_args)
        new_expr = cancel(expr.func(*largs))
        return new_expr
