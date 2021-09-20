from ..constants import g, c_p_dry, sigma
from .convective_adjustment import convective_adjustment
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import inspect

import numpy as np
def t_years_days(t):
    t_full_days = t / (24 * 60 ** 2)
    t_years, t_days = divmod(t_full_days, 365)
    return t_years, t_days


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


def get_isothermal_temp(albedo, F_stellar=None, latitude=None, T_star=None, R_star=None, star_planet_dist=None):
    if F_stellar is None:
        F_stellar = sigma * T_star ** 4 * R_star ** 2 / star_planet_dist ** 2
    if latitude is not None:
        F_stellar = F_stellar * latitudinal_solar_distribution(latitude)
    return np.power(F_stellar / sigma * (1 - albedo) / 4, 1 / 4)


def grid_points_near_local_maxima(q, q_local_maxima_index, nz_multiplier_param,
                                  q_thresh_info_percentile, q_thresh_info_max, nz):
    """
    Finds suitable number of grid points to give good resolution in q i.e. large number of points where
    changing fast near local maxima. p_array_indices are all indices of p needed to resolve q well.

    :param q: numpy array [p_initial_size]
        some form of dtau/dp
    :param q_local_maxima_index: list
        indices of local maxima in q
    :param nz_multiplier_param: float
        nz_multiplier_param * q_local_max_value is number of points about each local maxima.
    :param q_thresh_info_percentile: float
        region about local maxima in which to build grid ends when q falls below this percentile of q
    :param q_thresh_info_max: float
        region about local maxima in which to build grid ends when q falls below q_local_max_value / q_thresh_info_max
    :param nz: 'auto' or integer
        desired number of grid points
    :return: p_array_indices, nz. nz will be changed if was 'auto'.
    """
    p_initial_size = np.size(q)
    last_above_ind = 0  # use max (new_below_ind, above_ind) so local maxima sections don't overlap
    nLocalMaxima = len(q_local_maxima_index)
    q_max_values = q[q_local_maxima_index]
    if nz == 'auto':
        nz_multiplier = max(
            [nz_multiplier_param, max(5 / q_max_values)])  # have at least 5 grid points for each local maxima
        nPointsPerSet = np.ceil(q_max_values * nz_multiplier).astype(int)
        nz = sum(nPointsPerSet)
    else:
        nz_multiplier = None
        nPointsPerSet = np.floor(q_max_values / sum(q_max_values) * nz).astype(int)
        nPointsPerSet[-1] = nz - sum(nPointsPerSet[:-1])
    p_array_indices = []
    cum_q = np.cumsum(q)
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
    return p_array_indices, nz_multiplier, nz


def amend_sparse_pressure_grid(to_correct, nz_multiplier, p0, p_interface, q, target_log_delta_p, nz, ny):
    """
    Amends grid to desired pressure resolution in regions where grid points too sparse.

    :param to_correct: numpy array
        indices in p_interface where grid is too sparse
    :param nz_multiplier: float or None
        nz_multiplier * q_local_max_value is number of points about each local maxima.
    :param p0: numpy array [p_initial_size]
        initial very high resolution pressure grid
    :param p_interface: numpy array [nz]
        current pressure grid
    :param q: numpy array [p_initial_size]
        some form of dtau/dp
    :param target_log_delta_p: float
        Try to have a minimum log10 pressure separation of target_log_delta_p in grid.
    :param nz: integer
    :param ny: integer
    :return:
    """
    log_p = np.log10(p_interface)
    for i in to_correct:
        if nz_multiplier is not None:
            p_range = np.logical_and(p0 < p_interface[i], p0 > p_interface[i + 1])
            n_new_levels = max(int(max(q[p_range]) * nz_multiplier), 3)
            new_levels = np.logspace(log_p[i], log_p[i + 1], n_new_levels + 2)
            p_interface = np.flip(np.sort(np.append(p_interface, new_levels[1:-1])))
            nz = len(p_interface)
        else:
            n_new_levels = int(min(max(np.ceil((log_p[i - 1] - log_p[i]) / target_log_delta_p), 3), nz / 10))
            max_i_to_change = int(min(i + np.ceil(n_new_levels / 2), nz) - 1)
            min_i_to_change = int(max(max_i_to_change - n_new_levels, 0))
            if min_i_to_change == 0:
                max_i_to_change = n_new_levels
            new_levels = np.logspace(log_p[min_i_to_change], log_p[max_i_to_change], n_new_levels + 1)
            p_interface[min_i_to_change:max_i_to_change + 1] = new_levels

    p_interface = np.tile(p_interface, (ny, 1)).transpose()  # nz x ny
    return p_interface, nz


class Atmosphere:
    def __init__(self, nz, ny, F_stellar_constant, albedo=0.3, temp_change=1, delta_temp_change=0.01):
        """

        :param nz: integer or 'auto'.
            Number of pressure levels. If 'auto', chooses an appropriate amount such that have good spread
            in both pressure and optical thickness, tau
        :param ny: integer
            Number of latitudes
        :param F_stellar_constant: float
            Flux density at surface of planet. Units = W/m^2;
        :param albedo: list, optional.
            Upward short wave flux at top of atmosphere is (1-albedo[i])*F_stellar_constant*latitude_dist_factor[i]/4;
            at each latitude index i. If give single value, will repeat this value at each latitude.
            Can also be function of latitude.
            default: 0.3 at each latitude.
        :param temp_change: float, optional.
            Time step is found so that at least one level will have a temperature change equal to this.
            default: 1K.
        :param delta_temp_change: float, optional.
            If not converging, temp_change will be lowered by delta_temp_change.
            default: 0.001K.
        """
        self.nz = nz
        self.ny = ny
        self.latitude = np.linspace(-90, 90, self.ny)
        if inspect.isfunction(albedo):
            self.albedo = albedo(self.latitude)
        else:
            if np.size(albedo) < self.ny and np.size(albedo) == 1:
                self.albedo = np.repeat(albedo, self.ny)
            else:
                self.albedo = albedo
        self.F_stellar_constant = F_stellar_constant
        self.solar_latitude_factor = latitudinal_solar_distribution(self.latitude)
        self.T0 = get_isothermal_temp(self.albedo, self.F_stellar_constant, self.latitude)
        # Below is dict to track time step
        # have one temperature change of DeltaT K at each step.
        # nSameMaxInd and nSameMaxInd2 are used to find levels that are stagnating or oscillating.
        # These are then added to RemoveInd so will no longer update.
        self.time_step_info = {'DeltaT': temp_change, 'MaxTend': 0, 'MaxTendInd': -1,
                               'DeltaT_step': delta_temp_change, 'dt': 0,
                               'MaxDeltaT': temp_change, 'nSameMaxInd': 0, 'nSameMaxInd2': 0,
                               'RemoveInd': []}

    def update_temp(self, t, net_flux, changing_tau=False, convective_adjust=False,
                    net_flux_thresh=1e-7, net_flux_percentile=95):
        """
        This finds a suitable time step through the function update_time_step.
         It will then update the temperature profile accordingly.

        :param t: float.
            time since start of simulation.
        :param net_flux: numpy array [nz x ny]
            net flux at each pressure and latitude level for current iteration.
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
        # Use finite volume method
        T_tendency = g / c_p_dry * (
                net_flux[1:, :] - net_flux[:-1, :]) / (
                             self.p_interface[1:, :] - self.p_interface[:-1, :])
        if t > 0 and changing_tau is False:
            # Remove grid levels which are stagnating to reach equilibrium more quickly.
            levels_to_update = np.where(abs(net_flux[:-1].flatten()) > net_flux_thresh)[0]
            levels_to_update = np.setdiff1d(levels_to_update, self.time_step_info['RemoveInd'])
            delta_net_flux = np.percentile(abs(net_flux - self.net_flux), net_flux_percentile)
        else:
            # always update every level.
            levels_to_update = np.arange((self.nz - 1) * self.ny)
            delta_net_flux = 1e6  # any large number

        if len(levels_to_update) > 0:
            self.update_time_step(T_tendency, levels_to_update)
        self.net_flux = net_flux
        update_ind = np.unravel_index(levels_to_update, self.T.shape)  # from 1d to 2d
        self.T[update_ind] = self.T[update_ind] + self.time_step_info['dt'] * T_tendency[update_ind]
        if convective_adjust:
            self.T = convective_adjustment(self.p[:, 0], self.T)
        t = t + self.time_step_info['dt']
        return t, delta_net_flux

    def update_time_step(self, T_tendency, levels_to_update):
        """
        This finds the time step such that precisely one pressure level will have its temperature changed by
        self.time_step_info['DeltaT'] kelvin. This defaults to 1 kelvin and will be lowered if not converging.
        Info used to compute time step is stored in self.time_step_info.

        :param T_tendency: numpy array.
            Rate at which temperature is changing at each level. Units = K/s
        :param levels_to_update: list.
            Temperature is only being updated at grid levels indicated by this.
        """
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
                self.time_step_info['nSameMaxInd'] > 20 and self.time_step_info['nSameMaxInd2'] == 0) or (
                len(self.time_step_info['RemoveInd']) > 3 and self.time_step_info['nSameMaxInd'] +
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

    def evolve_to_equilibrium(self, data_dict=None, flux_thresh=1e-3, T_initial=None, convective_adjust=False,
                              save=True):
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
        i=0
        #print("Trying to reach equilibrium (flux_thresh = {:.4f})...".format(flux_thresh))
        while not equilibrium:
            t, delta_net_flux = self.take_time_step(t, T_initial, changing_tau=False,
                                                    convective_adjust=convective_adjust)
            if save:
                data_dict = self.save_data(data_dict, t)
            if i == 1:
                flux_thresh = min(flux_thresh, 0.99*delta_net_flux)
                print("Trying to reach equilibrium (flux_thresh = {:.4f})...".format(flux_thresh))
            equilibrium = self.check_equilibrium(delta_net_flux, flux_thresh)
            if min(self.T.flatten()) < 0:
                raise ValueError('Temperature is below zero')
            t_years, t_days = t_years_days(t)
            print("{:.0f} Years, {:.0f} Days: delta_net_flux = {:.4f}".format(t_years, t_days, delta_net_flux),
                  end="\r")
            i += 1
        print("{:.0f} Years, {:.0f} Days: delta_net_flux = {:.4f}".format(t_years, t_days, delta_net_flux))
        print("Done!")
        # set RemoveInd empty so will evolve all pressure levels if continue after this
        self.time_step_info['RemoveInd'] = []
        return data_dict

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
            t_years, t_days = t_years_days(t_plot[i])
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

        return anim
