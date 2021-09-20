from Model.radiation.real_gas import RealGas
import matplotlib.pyplot as plt
import matplotlib
from scipy import optimize
matplotlib.use('TkAgg')

"""CO2 only world"""
molecule_names = ['CO2']


gas = RealGas(nz=50, ny=1, molecule_names=molecule_names)
T_g = gas.find_Tg()
# T_g = 265
gas = RealGas(nz=50, ny=1, molecule_names=molecule_names, T_g=T_g)
flux_dict = {'lw_up': [], 'lw_down': [], 'sw_up': [], 'sw_down': []}
data = {'t': [], 'T': [], 'flux': flux_dict}
data = gas.save_data(data, 0)
data = gas.evolve_to_equilibrium(data, flux_thresh=1e-3)
anim = gas.plot_animate(data['T'], data['t'], flux_array=data['flux'], nPlotFrames=70)
plt.show()
