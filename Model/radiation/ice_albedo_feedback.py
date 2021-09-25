import numpy as np
from Model.radiation.grey import GreyGas
from ..constants import p_surface_earth, p_toa_earth
from .convective_adjustment import nearest_value_in_array
from tqdm import tqdm
import inspect
import matplotlib.pyplot as plt
from scipy.signal import convolve
import os
import contextlib


def albedo_step_function(latitude, T_surface=None, albedo_no_ice=0.3, albedo_ice=0.6, T_ice=263):
    """
    example function that returns albedo at each latitude which depends on surface temperature
    at that latitude.

    :param latitude: numpy array [ny x 1]
        array of latitudes in degrees
    :param T_surface: numpy array [ny x 1], optional.
        surface temperature at each latitude in Kelvin.
        default: None so no temperature influence on albedo.
    :param albedo_no_ice: float, optional.
        albedo at latitudes where there is no ice.
        default: 0.3
    :param albedo_ice: float, optional.
        albedo at latitudes where there is ice.
        default: 0.6
    :param T_ice: float, optional.
        Temperature in Kelvin below which the albedo is set to albedo_ice.
        default: 263
    :return:
    """
    albedo = np.ones_like(latitude) * albedo_no_ice
    if T_surface is not None:
        albedo[T_surface <= T_ice] = albedo_ice
    return albedo


class GreyAlbedoFeedback:
    def __init__(self, tau_lw_surface_values, stellar_constant_values,
                 nz, ny, tau_lw_func, tau_lw_func_args, tau_sw_func=None,
                 tau_sw_func_args=None, albedo=albedo_step_function,
                 p_surface=p_surface_earth, p_toa=p_toa_earth):
        """

        :param tau_lw_surface_values: numpy array or float
            optical depth in lon wave band at surface of planet.
            Either single number in which case stellar_constant_values must be an array
            or a numpy array.
        :param stellar_constant_values: numpy array or float
            Flux density at surface of planet. Units = W/m^2;
            Either single number in which case tau_lw_surface_values must be an array
            or a numpy array.
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
        :param albedo: function, optional.
            function that takes latitude and surface temperature as 1st and 2nd arguments and returns
            array of albedo values at each latitude. Function must also have albedo_no_ice, albedo_ice and T_ice
            as default arguments.
            default: albedo_step_function
        :param p_surface: float, optional.
            Pressure in Pa at surface.
            default: p_surface_earth = 101320 Pa
        :param p_toa: float, optional.
            Pressure in Pa at top of atmosphere
            default: p_toa_earth = 20 Pa
        """

        # albedo function must have albedo_no_ice, albedo_ice and T_ice as default arguments. Obtain them.
        signature = inspect.signature(albedo)
        self.albedo_function = albedo
        self.albedo_no_ice = signature.parameters['albedo_no_ice'].default
        self.albedo_ice = signature.parameters['albedo_ice'].default
        self.T_ice = signature.parameters['T_ice'].default

        # see if changing tau or F_stellar
        if np.size(tau_lw_surface_values) > 1 and np.size(stellar_constant_values) == 1:
            self.changing_param = 'tau'
        elif np.size(stellar_constant_values) > 1 and np.size(tau_lw_surface_values) == 1:
            self.changing_param = 'stellar'
        else:
            raise ValueError('Must have either tau_lw_surface_values or '
                             'stellar_constant_values be varying and the other constant')

        # start with largest value in each case, then work to smallest before increasing back to largest
        if self.changing_param == 'tau':
            self.changing_param_values = np.concatenate((np.sort(tau_lw_surface_values)[::-1],
                                                         np.sort(tau_lw_surface_values)[1:]))
            F_stellar_constant = stellar_constant_values
            self.tau_args = tau_lw_func_args.copy()
            self.tau_args[1] = self.changing_param_values[0]
        elif self.changing_param == 'stellar':
            self.changing_param_values = np.concatenate((np.sort(stellar_constant_values)[::-1],
                                                         np.sort(stellar_constant_values)[1:]))
            F_stellar_constant = self.changing_param_values[0]

        # Initialise gas with no ice albedo as start with warmest scenario which we assume is ice free.
        self.grey_world = GreyGas(nz, ny, tau_lw_func, tau_lw_func_args,
                                  tau_sw_func, tau_sw_func_args, F_stellar_constant, self.albedo_no_ice,
                                  p_surface=p_surface, p_toa=p_toa)

        # Get plotting latitude which includes 0
        if 0 in self.grey_world.latitude:
            self.latitude_plot = self.grey_world.latitude
        else:
            # if no 0 in latitudes, take average between adjacent latitudes
            # i.e. indicates lower latitude of each latitude bin in northern hemisphere.
            # add extra zero to deal with equator.
            self.latitude_plot = convolve(self.grey_world.latitude, np.ones(2), 'valid') / 2
            self.latitude_plot = np.concatenate((self.latitude_plot, [0]))
            self.latitude_plot = np.sort(self.latitude_plot)


    def update_albedo(self, delta_albedo=0.1, delta_net_flux_thresh=1e-3, conv_adjust=False):
        """
        This runs to equilibrium using last albedo values.
        Then checks surface temperature to see if any albedo values need changing.
        If they do, they are changed by delta_albedo until they reach their final no_ice or ice values.

        :param delta_albedo: float, optional.
            albedo increases/decreases in these increments as implemented by update_albedo.
            lower means slower to run but more stable.
            default: 0.1
        :param delta_net_flux_thresh: float, optional.
            Threshold in delta_net_flux to achieve equilibrium
            default: 1e-3
        :param convective_adjust: boolean, optional.
            Whether the temperature profile should adjust to stay stable with respect to convection.
            default: False
        """
        # Find temperature profile using last albedo values.
        albedo_last = self.grey_world.albedo.copy()
        _ = self.grey_world.evolve_to_equilibrium(flux_thresh=delta_net_flux_thresh,
                                                  convective_adjust=conv_adjust)
        # Find new albedo values with current temperature profile
        albedo_new = self.albedo_function(self.grey_world.latitude, self.grey_world.T[0, :])
        update_albedo_values = np.where(albedo_last != albedo_new)[0]
        delta_albedo_array = np.sign(albedo_new-albedo_last)[update_albedo_values] * delta_albedo
        while len(update_albedo_values) > 0:
            # incrementally update albedo values until reach final values
            self.grey_world.albedo[update_albedo_values] = np.clip(
                self.grey_world.albedo[update_albedo_values] + delta_albedo_array, self.albedo_no_ice, self.albedo_ice)
            _ = self.grey_world.evolve_to_equilibrium(flux_thresh=delta_net_flux_thresh,
                                                      convective_adjust=conv_adjust)
            update_albedo_values = np.where(self.grey_world.albedo != albedo_new)[0]

    def run(self, delta_albedo=0.1, delta_net_flux_thresh=1e-3, conv_adjust=False, print_eqb_info=False):
        """
        At each value of self.changing_param, we run to the equilibrium temperature profile.
        Then save the albedo, ice latitude and surface temperatures

        :param delta_albedo: float, optional.
            albedo increases/decreases in these increments as implemented by update_albedo.
            lower means slower to run but more stable.
            default: 0.1
        :param delta_net_flux_thresh: float, optional.
            Threshold in delta_net_flux to achieve equilibrium
            default: 1e-3
        :param conv_adjust: boolean, optional.
            Whether the temperature profile should adjust to stay stable with respect to convection.
            default: False
        :param print_eqb_info: boolean, optional.
            Whether to print approach to equilibrium info for each iteration of ice-albedo experiment.
            default: False
        :return:
            albedo_array, ice_latitude, T_surface
        """
        albedo_array = []
        ice_latitude = []
        T_surface = []
        for i in tqdm(range(0, len(self.changing_param_values))):

            if self.changing_param == 'tau':
                self.tau_args[1] = self.changing_param_values[i]
                self.grey_world.tau_lw_func_args = tuple(self.tau_args)
                self.grey_world.update_grid()
            elif self.changing_param == 'stellar':
                self.grey_world.F_stellar_constant = self.changing_param_values[i]
            if print_eqb_info:
                self.update_albedo(delta_albedo, delta_net_flux_thresh, conv_adjust)
            else:
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    self.update_albedo(delta_albedo, delta_net_flux_thresh, conv_adjust)
            albedo_array.append(self.grey_world.albedo.copy())
            ice_latitude.append(
                min(np.concatenate((abs(self.latitude_plot)[self.grey_world.albedo == self.albedo_ice], [90]))))
            T_surface.append(self.grey_world.T[0, :].copy())
        return albedo_array, ice_latitude, T_surface

    def plot(self, ice_latitude, T_surface, T_latitude=52.4):
        T_latitude = nearest_value_in_array(self.grey_world.latitude, T_latitude)
        T_latitude_index = np.where(self.grey_world.latitude == T_latitude)[0][0]
        T_surface = np.array(T_surface)
        ice_latitude = np.array(ice_latitude)
        T_surface_plot = T_surface[:, T_latitude_index]

        cooling_lim = self.changing_param_values.argmin() + 1
        cooling_indices = np.arange(cooling_lim)
        warming_indices = np.arange(cooling_lim - 1, len(self.changing_param_values))

        cooling_color = 'red'
        warming_color = 'blue'
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
        axs[0].plot(self.changing_param_values[cooling_indices], ice_latitude[cooling_indices], color=cooling_color,
                    label='cooling')
        axs[0].plot(self.changing_param_values[warming_indices], ice_latitude[warming_indices], color=warming_color,
                    label='warming')
        axs[0].legend()
        axs[0].set_ylabel('Ice edge latitude')
        axs[0].set_ylim((-5, 95))
        axs[1].plot(self.changing_param_values[cooling_indices], T_surface_plot[cooling_indices], color=cooling_color)
        axs[1].plot(self.changing_param_values[warming_indices], T_surface_plot[warming_indices], color=warming_color)
        axs[1].axhline(y=self.T_ice, color='k', linestyle=':', label=r'$T_{ice}$')
        axs[1].legend()
        axs[1].set_ylabel('$T_{surface}$ (K) at ' + str(round(T_latitude)) + '$^{\circ}$ latitude')
        if self.changing_param == 'tau':
            axs[1].set_xlabel(r'Long Wave Surface Optical Depth, $\tau_{lw, surface}$')
        elif self.changing_param == 'stellar':
            axs[1].set_xlabel(r'Stellar Constant, $F^{\odot}$ (Wm$^{-2}$)')
