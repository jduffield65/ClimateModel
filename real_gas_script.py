from Model.radiation.real_gas import RealGas
import matplotlib.pyplot as plt
import matplotlib
import Model.radiation.real_gas_data.hitran as hitran
from scipy import optimize
import numpy as np
matplotlib.use('TkAgg')

"""Making hitran absorption coef data"""
# # get data for toy gas with just one wide strong single line centered at peak of black body spectrum
# single_line_data = {'nu': np.array([520]), 'sw': np.array([4850.0]), 'gamma_air': np.array([0.1]),
#                     'n_air': np.array([0.7])}
# hitran.make_table(single_line_data, wavenumber_array=np.arange(320, 722, 10, dtype=float))
# hitran.make_table('CH4')
# ax = hitran.plot_absorption_coefficient('single_line', hitran.p_reference, 270)
# plt.show()
"""CO2 only world"""
molecule_names = ['CH4']


# gas = RealGas(nz=50, ny=1, molecule_names=molecule_names)
# T_g = gas.find_Tg()
T_g = 265
gas = RealGas(nz=50, ny=1, molecule_names=molecule_names, T_g=T_g)
flux_dict = {'lw_up': [], 'lw_down': [], 'sw_up': [], 'sw_down': []}
data = {'t': [], 'T': [], 'flux': flux_dict}
data = gas.save_data(data, 0)
data = gas.evolve_to_equilibrium(data, flux_thresh=1e-3)
ax = gas.plot_olr()
anim = gas.plot_animate(data['T'], data['t'], flux_array=data['flux'], nPlotFrames=70)
plt.show()
