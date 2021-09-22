from Model.radiation.real_gas import RealGas
import Model.radiation.real_gas_data.specific_humidity as humidity
import matplotlib.pyplot as plt
import matplotlib
import Model.radiation.real_gas_data.hitran as hitran
from scipy import optimize
import numpy as np
matplotlib.use('TkAgg')

conv_adjust = False
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
"""CO2 only world"""
molecule_names = ['CO2', 'CH4', 'H2O', 'O3']


# gas = RealGas(nz=50, ny=1, molecule_names=molecule_names, temp_change=1, delta_temp_change=0.1)
# T_g = gas.find_Tg(convective_adjust=conv_adjust)
T_g = 305
gas = RealGas(nz='auto', ny=1, molecule_names=molecule_names, T_g=T_g, p_toa=0.1, temp_change=1, delta_temp_change=0.1)
# # different for single_line as specify q
# gas = RealGas(nz='auto', ny=1, molecule_names=molecule_names, T_g=T_g, q_funcs={'single_line': humidity.co2},
#               q_funcs_args={'single_line': ()})
flux_dict = {'lw_up': [], 'lw_down': [], 'sw_up': [], 'sw_down': []}
data = {'t': [], 'T': [], 'flux': flux_dict}
data = gas.save_data(data, 0)
data = gas.evolve_to_equilibrium(data, flux_thresh=1e-3, convective_adjust=conv_adjust)
ax = gas.plot_olr()
ax2 = gas.plot_incoming_short_wave()
anim = gas.plot_animate(data['T'], data['t'], flux_array=data['flux'], nPlotFrames=70)
plt.show()
