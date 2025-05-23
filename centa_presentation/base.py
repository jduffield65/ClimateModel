# import warnings
# warnings.filterwarnings("ignore") # ignore all warnings in this notebook
import warnings

warnings.filterwarnings("ignore")  # ignore all warnings in this notebook
import sys

sys.path.append("/Users/joshduffield/Documents/PlanetaryClimate/ClimateModel")
import numpy as np
import Model.radiation.real_gas_data.hitran as hitran
import Model.radiation.real_gas_data.specific_humidity as humidity
import Model.radiation.real_gas as rg
import matplotlib.pyplot as plt


def plot_T_q(atmos, log_q=True):
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    axs[0].get_shared_y_axes().join(axs[0], axs[1])
    axs[0].plot(atmos.T, atmos.p)
    axs[0].axes.invert_yaxis()
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Pressure / Pa')
    axs[0].set_xlabel('Temperature / K')
    for molecule_name in atmos.molecule_names:
        if molecule_name == 'single_line':
            M_name = 'CO2'
        else:
            M_name = molecule_name
        axs[1].plot(humidity.ppmv_from_humidity(
            atmos.q_funcs[molecule_name](atmos.p, *atmos.q_funcs_args[molecule_name]), M_name),
            atmos.p, label=molecule_name)
    if log_q:
        axs[1].set_xscale('log')
    axs[1].set_xlabel('Volume Mixing Ratio (ppmv)')
    axs[1].legend()


def update_tau(atmos, q_args, T_func):
    """
    updates optical depth interface, tau_interface of atmos using new molecule humidity distribution given by q_args

    :param atmos: RealGas class
    :param q_args: dictionary, humidity arguments for each molecule in atmos
    :param T_func: function which takes pressure as input and returns temperature at that pressure
    :return:
    """
    atmos.q_funcs_args = q_args
    T_interface = T_func(atmos.p_interface)
    atmos.tau_interface = rg.optical_depth(atmos.p_interface[:, 0], T_interface, atmos.nu,
                                           atmos.molecule_names, atmos.q_funcs, atmos.q_funcs_args)


def update_flux(atmos, q_args, T_func):
    """

    :param atmos: RealGas class
    :param q_args:
    :param T_func: function which takes pressure as input and returns temperature at that pressure
    :return:
    """
    update_tau(atmos, q_args, T_func)
    atmos.up_flux, atmos.down_flux = atmos.get_flux()
    atmos.net_flux = np.sum(atmos.up_flux * atmos.nu_bands['delta'], axis=1) - \
                     np.sum(atmos.down_flux * atmos.nu_bands['delta'], axis=1)


def eqv_ppmv(molecule, co2_ppmv):
    """
    gets mass of molecule equivalent to adding co2_ppmv over whole atmosphere.

    :param molecule: string, molecule name
    :param co2_ppmv: float, amount of co2 added in ppmv
    :return: float
    """
    mass_co2_added = co2_ppmv * humidity.molecules['CO2']['M']
    eqv_ppmv_molecule_added = mass_co2_added / humidity.molecules[molecule.upper()]['M']
    return eqv_ppmv_molecule_added


def get_olr_area(atmos, flux=None):
    """
    gets area under OLR curve for given atmosphere

    :param atmos:
    :return:
    """
    max_nu_band = [nu_band[1] for nu_band in atmos.nu_bands['range']]
    lw_band = max_nu_band <= atmos.nu_lw.max()
    if flux is None:
        flux = atmos.up_flux[0]
    area_under_curve = np.trapz(flux[lw_band], atmos.nu_bands['centre'][lw_band])
    return area_under_curve


def get_olr_area_add_ghg(atmos, ghg_molecule, co2_ppmv_added, T_func):
    """
    gets area under OLR curve for each mass of ghg added. Requires constant humidity profile of ghg_molecule

    :param atmos: RealGas object
    :param ghg_molecule: string, ghg molecule to change concentration of and see effect on OLR
        Has to have constant_q humidity profile
    :param co2_ppmv_added: float array [n_conc] must start with 0. mass of ghg in co2 equivalent e.g. [0, 5, 10, 15]
    :param T_func: function which takes pressure as input and returns temperature
    :return: float array [n_conc] total OLR at each GHG concentration.
    surface_flux is contribution just from flux coming from surface which has not been absorbed by atmosphere
    """
    q_args_base = atmos.q_funcs_args.copy()
    if co2_ppmv_added[0] != 0:
        raise ValueError("co2_ppmv_added should have 0 as the first value as we are interested in OLR reduction.")
    if not isinstance(q_args_base[ghg_molecule.upper()][1], str):
        raise ValueError(f"{ghg_molecule} should have a constant_q specific humidity profile")
    tot_flux = []
    surface_flux = []
    for co2_ppmv in co2_ppmv_added:
        q_args = q_args_base.copy()
        q_args[ghg_molecule.upper()] = (q_args_base[ghg_molecule.upper()][0] + eqv_ppmv(ghg_molecule.upper(), co2_ppmv),
                                        ghg_molecule.upper())
        update_flux(atmos, q_args, T_func)
        tot_flux.append(get_olr_area(atmos))
        surface_flux.append(get_surface_up_flux_olr_area(atmos))
    return np.array(tot_flux), np.array(surface_flux)


def ghg_diff_initial_h2o_plot(ax, atmos, h2o_scale_factors, ghg_molecule, co2_ppmv_added, T_func):
    """
    plots how OLR changes with increasing greenhouse gas concentration for different h2o initial concentrations
    Assumes ghg_molecule has constant_q humidity profile

    :param ax: axis of figure to plot on
    :param atmos: RealGas object
    :param h2o_scale_factors: float array e.g. [1, 0.5, 0.01]
        what to multiply H2O specific humidity profile by
    :param ghg_molecule: string, ghg molecule to change concentration of and see effect on OLR
        Has to have constant_q humidity profile
    :param co2_ppmv_added: float array must start with 0. mass of ghg in co2 equivalent e.g. [0, 5, 10, 15]
    :param T_func: function which takes pressure as input and returns temperature
    :return: ax
    """
    q_args_base_h2o_base_ghg = atmos.q_funcs_args.copy()
    for h2o_scale in h2o_scale_factors:
        q_args_base_ghg = q_args_base_h2o_base_ghg.copy()
        q_args_base_ghg['H2O'] = (h2o_scale,)
        atmos.q_funcs_args = q_args_base_ghg
        tot_flux = get_olr_area_add_ghg(atmos, ghg_molecule, co2_ppmv_added, T_func)
        if h2o_scale < 1 and h2o_scale != 0:
            label = "{:.1f}".format(h2o_scale)
        else:
            label = "{:.0f}".format(h2o_scale)
        ax.plot(co2_ppmv_added, tot_flux - tot_flux[0], label=label)
    update_flux(atmos, q_args_base_h2o_base_ghg, T_func)  # return to initial value
    ax.legend(title="Multiple of\n$H_2O$ concentration")
    ax.set_ylabel(r"$\Delta OLR$ (W/m$^2$)")
    ax.set_xlabel(f"Mass of {ghg_molecule} added ($CO_2$ ppmv)")
    return ax


def get_ghg_activity(atmos, molecule=None):
    """
    gets the absolute rate of change of transmission from surface to top of atmosphere per mass equivalent to
    1ppmv CO2 added of a molecule with k=1. Assumes constant_q profile of molecule_name in atmosphere.
    if None, just sets absorb_coef to 1 for all wavenumbers

    :param atmos:
    :param molecule:
    :return:
    median wavenumber of each band
    -dtransmission_dq corresponding to each wavenumber band
    """
    if molecule is None:
        absorb_coef = np.ones_like(atmos.nu)
    else:
        absorb_coef = rg.load_absorption_coef(np.array([atmos.p_surface]),
                                              np.array([atmos.T_g]), atmos.nu, molecule)
        absorb_coef = absorb_coef.flatten()
    p1 = atmos.p_interface[0]  # toa
    p2 = atmos.p_interface[-1]  # surface
    nu_centres = atmos.nu_bands['centre'][~atmos.nu_bands['sw']]
    nu_ranges = np.array(atmos.nu_bands['range'])[~atmos.nu_bands['sw']]
    nu_deltas = atmos.nu_bands['delta'][~atmos.nu_bands['sw']]
    dtrans_dq = np.zeros_like(nu_centres)
    # mass_conv so get change in transmission per mass equivalent to 1 ppmv CO2 added
    mass_conv = humidity.humidity_from_ppmv(1, 'CO2')
    for i in range(dtrans_dq.shape[0]):
        dtrans_dq[i] = rg.dtransmission_dq(p1, p2, atmos.p_interface, nu_ranges[i], nu_deltas[i],
                                           atmos.nu, atmos.tau_interface, absorb_coef)
    return nu_centres, -dtrans_dq * mass_conv


def get_surface_up_flux_olr_area(atmos):
    """
    gets up_flux resulting only from attenuation of surface flux

    :param atmos:
    :return:
    """
    up_flux = np.ones((atmos.nz, atmos.n_nu_bands)) * np.pi * rg.B_wavenumber(atmos.nu_bands['centre'], atmos.T_g)
    for j in range(atmos.n_nu_bands):
        # Apply exponential decay of surface flux at lower pressures
        up_flux[:, j] = up_flux[:, j] * rg.transmission(atmos.p_interface[:, 0], atmos.p_interface[-1:, 0],
                                                        atmos.p_interface[:, 0], atmos.nu_bands['range'][j],
                                                        atmos.nu_bands['delta'][j], atmos.nu,
                                                        atmos.tau_interface).reshape(-1)
    max_nu_band = [nu_band[1] for nu_band in atmos.nu_bands['range']]
    lw_band = max_nu_band <= atmos.nu_lw.max()
    area_under_curve = np.trapz(up_flux[0, lw_band], atmos.nu_bands['centre'][lw_band])
    return area_under_curve
