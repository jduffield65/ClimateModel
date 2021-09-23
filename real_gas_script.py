from Model.radiation.real_gas import RealGas, optical_depth
import Model.radiation.real_gas_data.specific_humidity as humidity
from Model.radiation.animation import Animate
import matplotlib.pyplot as plt
import matplotlib
import Model.radiation.real_gas_data.hitran as hitran
from scipy import optimize
import numpy as np
matplotlib.use('TkAgg')

conv_adjust = True
molecule_names = ['CO2', 'CH4', 'H2O', 'O3']

"""Making hitran absorption coef data"""
# # get data for toy gas with just one wide strong single line centered at peak of black body spectrum
# single_line_data = {'nu': np.array([520]), 'sw': np.array([500.0]), 'gamma_air': np.array([0.1]),
#                     'n_air': np.array([0.7])}
# hitran.make_table(single_line_data, wavenumber_array=np.arange(320, 722, 10, dtype=float),
#                   p_array=np.array([hitran.p_reference], dtype=float),
#                   T_array=np.array([hitran.T_reference], dtype=float))
# # hitran.make_table('CH4')
# ax = hitran.plot_absorption_coefficient('custom', hitran.p_reference, 270)
# plt.show()

"""Evolving CO2 conc - finding list of ground temp eqb"""
# gas = RealGas(nz=50, ny=1, molecule_names=molecule_names, temp_change=1, delta_temp_change=0.1)
# n_conc_values = 2
# q_args = [gas.q_funcs_args.copy()]
# co2_multiplier = 50.0
# T_g_values = []
# T_g_values.append(gas.find_Tg(convective_adjust=conv_adjust))
# for i in range(1, n_conc_values):
#     q_args.append(gas.q_funcs_args.copy())
#     q_args[i]['CO2'] = (list(q_args[i-1].copy()['CO2'])[0]*co2_multiplier, *q_args[i-1]['CO2'][1:])
#     gas = RealGas(nz=50, ny=1, molecule_names=molecule_names, temp_change=1, delta_temp_change=0.1,
#                   q_funcs_args=q_args[i])
#     T_g_values.append(gas.find_Tg(convective_adjust=conv_adjust))
#

""""""
# gas = RealGas(nz=50, ny=1, molecule_names=molecule_names, temp_change=1, delta_temp_change=0.1)
# T_g = gas.find_Tg(convective_adjust=conv_adjust)
T_g = 302.93
gas = RealGas(nz='auto', ny=1, molecule_names=molecule_names, T_g=T_g, p_toa=0.1, temp_change=1, delta_temp_change=0.1)
# # different for single_line as specify q
# gas = RealGas(nz='auto', ny=1, molecule_names=molecule_names, T_g=T_g, q_funcs={'single_line': humidity.co2},
#               q_funcs_args={'single_line': ()})
flux_dict = {'lw_up': [], 'lw_down': [], 'sw_up': [], 'sw_down': []}
# q_dict = {'CO2': [], 'CH4': [], 'H2O': [], 'O3': []}
data = {'t': [], 'T': [], 'flux': flux_dict}  # , 'q': q_dict}
data = gas.save_data(data, 0)
data = gas.evolve_to_equilibrium(data, flux_thresh=1e-3, convective_adjust=conv_adjust, t_end=2.0)
anim = Animate(gas, data['T'], data['t'], flux_array=data['flux'], nPlotFrames=70)  # , q_array=data['q'])
plt.show()
ax = gas.plot_olr()
ax2 = gas.plot_incoming_short_wave()
plt.show()
