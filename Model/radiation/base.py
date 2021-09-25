from ..constants import g, c_p_dry, sigma, p_surface_earth, p_toa_earth
from .convective_adjustment import convective_adjustment
import inspect
import numpy as np
from math import ceil, floor
import warnings
warnings.filterwarnings("ignore", message="Attempted to set non-positive bottom ylim on a log-scaled axis.")


def round_any(x, base, round_type='round'):
    """
    rounds the number x to the nearest multiple of base with the rounding done according to round_type.
    e.g. round_any(3, 5) = 5. round_any(3, 5, 'floor') = 0.
    """
    if round_type == 'round':
        return base * round(x / base)
    elif round_type == 'ceil':
        return base * ceil(x / base)
    elif round_type == 'floor':
        return base * floor(x / base)


def t_years_days(t):
    """given t in seconds, returns (t_years, t_days)"""
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
    """
    Computes the temperature of a planet with no atmosphere such that is in equilibrium with the incoming stellar
    radiation.
    :param albedo: numpy array [n_latitudes]
        fraction of stellar light reflected at each latitude.
    :param F_stellar: float
        Flux density at surface of planet. Units = W/m^2;
    :param latitude: numpy array [n_latitudes]
        array of latitudes in degrees. If don't provide, finds global average.
    :param T_star: float (not needed if give F_stellar)
        Temperature of star in (Kelvin)
    :param R_star: float (not needed if give F_stellar)
        Radius of star (m)
    :param star_planet_dist: float (not needed if give F_stellar)
        Average distance between star and planet (m)
    :return: [n_latitudes] numpy array
    """
    if F_stellar is None:
        F_stellar = sigma * T_star ** 4 * R_star ** 2 / star_planet_dist ** 2
    if latitude is not None:
        F_stellar = F_stellar * latitudinal_solar_distribution(latitude)
    return np.power(F_stellar / sigma * (1 - albedo) / 4, 1 / 4)


class Atmosphere:
    def __init__(self, nz, ny, F_stellar_constant, albedo=0.3, p_surface=p_surface_earth, p_toa=p_toa_earth,
                 temp_change=1, delta_temp_change=0.01):
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
        :param p_surface: float, optional.
            Pressure in Pa at surface.
            default: p_surface_earth = 101320 Pa
        :param p_toa: float, optional.
            Pressure in Pa at top of atmosphere
            default: p_toa_earth = 20 Pa
        :param temp_change: float, optional.
            Time step is found so that at least one level will have a temperature change equal to this.
            default: 1K.
        :param delta_temp_change: float, optional.
            If not converging, temp_change will be lowered by delta_temp_change.
            default: 0.01K.
        """
        self.nz = nz
        self.ny = ny
        self.p_surface = p_surface
        self.p_toa = p_toa
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
                    net_flux_thresh=1e-7, net_flux_percentile=95, conv_thresh=1e-5, conv_t_multiplier=5):
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
        :param conv_thresh: float, optional.
            if MaxTendInd is in convective region, multiply timestep by conv_t_multiplier
            convective if temperature difference between pre and post convective_adjustment is above conv_thresh
            default: 1e-5 K
        :param conv_t_multiplier: float, optional.
            default: 5
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
        if 'convective_levels' not in self.time_step_info:
            self.time_step_info['convective_levels'] = np.array([])
        if len(levels_to_update) > 0:
            self.update_time_step(T_tendency, levels_to_update)
            if convective_adjust and self.time_step_info['MaxTendInd'] in self.time_step_info['convective_levels']:
                self.time_step_info['dt'] = self.time_step_info['dt'] * conv_t_multiplier
        self.net_flux = net_flux
        update_ind = np.unravel_index(levels_to_update, self.T.shape)  # from 1d to 2d
        self.T[update_ind] = self.T[update_ind] + self.time_step_info['dt'] * T_tendency[update_ind]
        if convective_adjust:
            T_new = convective_adjustment(self.p[:, 0], self.T.copy())
            # ensure time step level not in convective region
            self.time_step_info['convective_levels'] = \
                levels_to_update[np.where(abs(T_new.flatten()[levels_to_update] - self.T.flatten()[levels_to_update])
                                          > conv_thresh)[0]]
            self.T = T_new
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

        # need 'nSameMaxInd'>1 for oscillation as if 'nSameMaxInd'==1 that could indicate a permanent
        # switch of direction.
        if (self.time_step_info['nSameMaxInd'] > 1 and self.time_step_info['nSameMaxInd2'] > 10) or (
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
                              save=True, t_end=4.0, conv_thresh=1e-5, conv_t_multiplier=5):
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
        :param save: boolean, optional.
            Whether to save data to dictionary or not
            default: True
        :param t_end: float, optional.
            Simulation will end after t_end years no matter what.
            default: 4 years
        :param conv_thresh: float, optional.
            if MaxTendInd is in convective region, multiply timestep by conv_t_multiplier
            convective if temperature difference between pre and post convective_adjustment is above conv_thresh
            default: 1e-5 K
        :param conv_t_multiplier: float, optional.
            default: 5
        :return:
            data_dict
        """
        if data_dict is None:
            if T_initial is None:
                T_initial = self.T.copy()
            data_dict = {'t': [0], 'T': [T_initial.copy()]}
        t = data_dict['t'][-1]
        t0 = t_years_days(t)[0] + t_years_days(t)[1]/365  # start time in years
        equilibrium = False
        i = 0
        while not equilibrium:
            t, delta_net_flux = self.take_time_step(t, T_initial, changing_tau=False,
                                                    convective_adjust=convective_adjust,
                                                    conv_thresh=conv_thresh, conv_t_multiplier=conv_t_multiplier)
            if save:
                data_dict = self.save_data(data_dict, t)
            if i == 1:
                flux_thresh = min(flux_thresh, 0.99*delta_net_flux)
                print("Trying to reach equilibrium (flux_thresh = {:.4f})...".format(flux_thresh))
            equilibrium = self.check_equilibrium(delta_net_flux, flux_thresh)
            if min(self.T.flatten()) < 0:
                raise ValueError('Temperature is below zero')
            t_years, t_days = t_years_days(t)
            if t_years + t_days/365 - t0 > t_end:
                equilibrium = True
            print("{:.0f} Years, {:.0f} Days: delta_net_flux = {:.4f}".format(t_years, t_days, delta_net_flux),
                  end="\r")
            i += 1
        print("{:.0f} Years, {:.0f} Days: delta_net_flux = {:.4f}".format(t_years, t_days, delta_net_flux))
        print("Done!")
        # set RemoveInd empty so will evolve all pressure levels if continue after this
        # Also reset nSameMaxInd to default values
        self.time_step_info['RemoveInd'] = []
        self.time_step_info['nSameMaxInd'] = 0
        self.time_step_info['nSameMaxInd2'] = 0
        self.time_step_info['MaxTendInd'] = -1 # so no change of nSameMaxInd on next iteration
        return data_dict
