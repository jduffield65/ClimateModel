import sys

sys.path.append("/Users/joshduffield/Documents/PlanetaryClimate/ClimateModel")
from base import *
import Model.radiation.real_gas_data.specific_humidity as humidity
from Model.radiation.real_gas_data.temperature_profiles import earth_temp
import Model.radiation.real_gas as rg
import matplotlib.pyplot as plt

co2_color = 'C0'
ch4_color = 'C1'
h2o_color = 'C2'
o3_color = 'C3'
potential_color = 'k'
potential_ylim = (1e-5, 0.1)
k_ylim = (1e-6, 10**4)
activity_ylim = (1e-5, 10)
wavenumber_lim = (400, 1600)
p_toa = 0.1  # Top of atmosphere pressure for this notebook (Pa)
n_bands = 200  # number of wavenumber bands in this notebook
q_funcs = {'CO2': humidity.constant_q, 'CH4': humidity.constant_q, 'H2O': humidity.h2o}
q_args = {'CO2': (370, 'CO2'), 'CH4': (1.75, 'CH4'), 'H2O': (0,)}
earth_atmos = rg.RealGas(nz='auto', ny=1, molecule_names=['CO2', 'CH4', 'H2O'], p_toa=p_toa,
                         T_func=earth_temp, n_nu_bands=n_bands, q_funcs=q_funcs, q_funcs_args=q_args)

### Temperature and humidity profiles
# plot_T_q(earth_atmos)
# plt.savefig('/Users/joshduffield/Documents/PlanetaryClimate/ClimateModel/presentation/figures/books_read.png',
#             bbox_inches='tight')
# plt.show()

### effect on OLR of changing CO2 and CH4 concentrations
q_args_base = earth_atmos.q_funcs_args.copy()
co2_mass_add = np.linspace(0.0, 92, 5)  # 92ppmv co2 added since industrial times
co2_tot_flux, co2_surface_flux = get_olr_area_add_ghg(earth_atmos, 'CO2', co2_mass_add, earth_temp)
update_flux(earth_atmos, q_args_base, earth_temp)  # return to initial value
ch4_ppmv_add = np.linspace(0.0, 1, 5)  # 1ppmv methane added since industrial times
# convert ppmv methane to ppmv CO2
ch4_mass_add = ch4_ppmv_add * humidity.molecules['CH4']['M'] / humidity.molecules['CO2']['M']
ch4_tot_flux, ch4_surface_flux = get_olr_area_add_ghg(earth_atmos, 'CH4', ch4_mass_add, earth_temp)
update_flux(earth_atmos, q_args_base, earth_temp)  # return to initial value
#
fig, ax = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
ax[0].set_ylabel(r"$\Delta OLR$ (W/m$^2$)")
ax[0].set_xlabel(f"Mass of $CO_2$ added ($CO_2$ ppmv)")
ax[0].plot(co2_mass_add, co2_tot_flux - co2_tot_flux[0], color=co2_color, label='Total')
ax[1].set_xlabel(f"Mass of $CH_4$ added ($CO_2$ ppmv)")
ax[1].plot(ch4_mass_add, ch4_tot_flux - ch4_tot_flux[0], color=ch4_color, label='Total')
ax[0].set_title("Changing $CO_2$")
ax[1].set_title("Changing $CH_4$")
ax[0].set_ylim(-1, 0)
ax[0].set_xlim(co2_mass_add[0], co2_mass_add[-1])
ax[1].set_xlim(ch4_mass_add[0], ch4_mass_add[-1])
ax[0].plot(co2_mass_add, co2_surface_flux - co2_surface_flux[0], color=co2_color, linestyle=':', label='Surface')
ax[1].plot(ch4_mass_add, ch4_surface_flux - ch4_surface_flux[0], color=ch4_color, linestyle=':', label='Surface')
ax[0].legend()
ax[1].legend()
plt.show()

### plot potential and activity for co2 and ch4 with earth composition
wave_number, ghg_potential = get_ghg_activity(earth_atmos)
absorb_coef_co2 = rg.load_absorption_coef(np.array([earth_atmos.p_surface]), np.array([earth_atmos.T_g]),
                                      wave_number, 'co2')
absorb_coef_ch4 = rg.load_absorption_coef(np.array([earth_atmos.p_surface]), np.array([earth_atmos.T_g]),
                                      wave_number, 'ch4')
absorb_coef_h2o = rg.load_absorption_coef(np.array([earth_atmos.p_surface]), np.array([earth_atmos.T_g]),
                                      wave_number, 'h2o')
absorb_coef_o3 = rg.load_absorption_coef(np.array([earth_atmos.p_surface]), np.array([earth_atmos.T_g]),
                                      wave_number, 'o3')
_, activity_co2 = get_ghg_activity(earth_atmos, 'CO2')
_, activity_ch4 = get_ghg_activity(earth_atmos, 'CH4')

fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
axs[0].plot(wave_number, ghg_potential, color=potential_color)
axs[0].tick_params(axis='y', labelcolor=potential_color)
axs[0].set_ylabel(r"Potential (kgm$^{-2}$ / $CO_2$ ppmv)", color=potential_color)
axs[0].set_xlim(wavenumber_lim)
axs[0].set_ylim(potential_ylim)
axs[0].set_yscale('log')
ax2 = axs[0].twinx()
ax2.set_ylabel(r"k (m$^2$kg$^{-1}$)", color=ch4_color)
ax2.plot(wave_number, absorb_coef_co2.flatten(), color=co2_color, label=r'$CO_2$')
ax2.plot(wave_number, absorb_coef_ch4.flatten(), color=ch4_color, label=r'$CH_4$')
ax2.tick_params(axis='y', labelcolor=co2_color)
ax2.set_ylim(k_ylim)
ax2.set_yscale('log')
ax2.legend()
axs[1].plot(wave_number, activity_co2, color=co2_color)
axs[1].plot(wave_number, activity_ch4, color=ch4_color)
axs[1].set_ylim(activity_ylim)
axs[1].set_yscale('log')
axs[1].set_ylabel(r"Activity (1 / $CO_2$ ppmv)", color=potential_color)
axs[1].set_xlabel(r"Wavenumber (cm$^{-1}$)")
plt.show()

### plot absorption coefficient
fig, ax2 = plt.subplots(1, 1, figsize=(20, 10))
ax2.set_ylabel(r"k (m$^2$kg$^{-1}$)")
ax2.plot(wave_number, absorb_coef_co2.flatten(), color=co2_color, label=r'$CO_2$')
ax2.plot(wave_number, absorb_coef_ch4.flatten(), color=ch4_color, label=r'$CH_4$')
ax2.plot(wave_number, absorb_coef_h2o.flatten(), color=h2o_color, label=r'$H_2O$')
ax2.set_ylim(k_ylim)
ax2.set_xlim(wavenumber_lim)
ax2.set_yscale('log')
ax2.legend()
ax2.set_xlabel(r"Wavenumber (cm$^{-1}$)")
plt.show()

### plot OLR vs mass ghg added for different H2O
h2o_scales = [1, 0.5, 0.1, 0]
ch4_ppmv_add = np.linspace(0.0, 1, 5)  # 1ppmv methane added since industrial times
# convert ppmv methane to ppmv CO2
ppmv_mass_add = ch4_ppmv_add * humidity.molecules['CH4']['M'] / humidity.molecules['CO2']['M']
fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
axs[0] = ghg_diff_initial_h2o_plot(axs[0], earth_atmos, h2o_scales, "CO2", ppmv_mass_add, earth_temp)
axs[1] = ghg_diff_initial_h2o_plot(axs[1], earth_atmos, h2o_scales, "CH4", ppmv_mass_add, earth_temp)
axs[0].set_title("Changing $CO_2$")
axs[1].set_title("Changing $CH_4$")
plt.show()
hi = 5

hi = 5
