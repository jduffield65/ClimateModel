import matplotlib.pyplot as plt
import Model.radiation.grey
import matplotlib
import numpy as np
from Model.constants import p_surface
from matplotlib.animation import FuncAnimation
from Model.radiation.grey import OpticalDepthFunctions
from Model.constants import p_surface

matplotlib.use('TkAgg')  # To make plot pop out

ny = 30
'''Analytic solution with short wave'''
# p_width_lw = 100000
# # alpha ratio must be integer for analytic solution
# alpha_sw = OpticalDepthFunctions.get_exponential_alpha(p_width_lw) / 5
# p_width_sw = OpticalDepthFunctions.get_exponential_p_width(alpha_sw)
# grey_world = Model.radiation.grey.GreyGas(nz='auto', ny=ny, tau_lw_func=OpticalDepthFunctions.exponential,
#                                           tau_lw_func_args=[p_width_lw, 4], tau_sw_func=OpticalDepthFunctions.exponential,
#                                           tau_sw_func_args=[p_width_sw, 0.6])
'''With stratosphere'''
# grey_world = Model.radiation.grey.GreyGas(nz='auto', ny=ny, tau_lw_func=OpticalDepthFunctions.exponential,
#                                           tau_lw_func_args=[100000, 4], tau_sw_func=OpticalDepthFunctions.peak_in_atmosphere,
#                                           tau_sw_func_args=[30000, 2000, 0.5])
'''With mesosphere'''
# grey_world = Model.radiation.grey.GreyGas(nz='auto',ny=ny, tau_lw_func=OpticalDepthFunctions.scale_height_and_peak_in_atmosphere,
#                                           tau_lw_func_args=[50000, 4, 1000, 600, 0.3], tau_sw_func=OpticalDepthFunctions.peak_in_atmosphere,
#                                           tau_sw_func_args=[10000, 2000, 0.05])
'''With thermosphere'''
grey_world = Model.radiation.grey.GreyGas(nz='auto', ny=ny, tau_lw_func=OpticalDepthFunctions.scale_height_and_peak_in_atmosphere,
                                          tau_lw_func_args=[51000, 4, 100, 600, 0.1], tau_sw_func=OpticalDepthFunctions.scale_height_and_peak_in_atmosphere,
                                          tau_sw_func_args=[p_surface, 0.12, 100, 20, 0.002])

""" Approach to equilibrium"""
# if grey_world.ny == 1:
#     up_flux_eqb, down_flux_eqb, T_eqb, up_sw_flux_eqb, down_sw_flux_eqb, \
#         correct_solution = grey_world.equilibrium_sol()
#     if correct_solution:
#         grey_world.plot_eqb(up_flux_eqb, down_flux_eqb, T_eqb, up_sw_flux_eqb, down_sw_flux_eqb)
# #plt.show()
# t = 0
# # Get temperature results until net_flux is zero everywhere i.e. equilibrium
# net_flux_thresh = 1e-1
# data = grey_world.evolve_to_equilibrium(flux_thresh=net_flux_thresh)
# if grey_world.ny == 1:
#     anim = grey_world.plot_animate(data['T'], data['t'], T_eqb, correct_solution)
# else:
#     anim = grey_world.plot_animate(data['T'], data['t'], nPlotFrames=30)
# plt.show()

""" Evolution with tau"""
p1 = OpticalDepthFunctions.get_exponential_p_width(1e-5)
p2 = OpticalDepthFunctions.get_exponential_p_width(1e-5 / 3, 2000)
tau_params_final = [100000, 6]
tau_params = [100000, 4]
p_max = 2000
tau_sw_params_final = [300000, 2000, 1.2]
tau_sw_params = [300000, 2000, 0]
grey_world = Model.radiation.grey.GreyGas(nz='auto', ny=ny, tau_lw_func=OpticalDepthFunctions.exponential,
                                          tau_lw_func_args=tau_params_final,
                                          tau_sw_func=OpticalDepthFunctions.peak_in_atmosphere,
                                          tau_sw_func_args=tau_sw_params_final)
grey_world.tau_lw_func_args = tuple(tau_params)
grey_world.tau_sw_func_args = tuple(tau_sw_params)
grey_world.update_grid()
up_flux_eqb, down_flux_eqb, T_eqb, up_sw_flux_eqb, down_sw_flux_eqb, \
correct_solution = grey_world.equilibrium_sol()
t = 0
t_end = 10 * 365 * 24 * 60 ** 2
t_sw = t_end
t_array = [0]
T_array = [T_eqb.copy()]
tau_array = {'lw': [grey_world.tau.copy()], 'sw': [grey_world.tau_sw.copy()]}
flux_array = {'lw_up': [up_flux_eqb.copy()], 'lw_down': [down_flux_eqb.copy()],
              'sw_up': [up_sw_flux_eqb.copy()], 'sw_down': [down_sw_flux_eqb.copy()]}
data = {'t': t_array, 'T': T_array, 'tau': tau_array, 'flux': flux_array}
changing_tau = True
delta_net_flux_thresh = 1e-3

while t < t_end:
    tau_params[1] = min(tau_params[1] + 1e-8 * t, tau_params_final[1])
    grey_world.tau_lw_func_args = tuple(tau_params)
    if tau_params[1] == tau_params_final[1] and tau_sw_params[2] != tau_sw_params_final[2]:
        if t_sw == t_end:
            # once lw optical depth reached max value, get to equilibrium
            data = grey_world.evolve_to_equilibrium(data, delta_net_flux_thresh, T_eqb.copy())
            t = data['t'][-1]
            t_sw = t
        # After equilibrium evolve sw optical depth
        tau_sw_params[2] = min(tau_sw_params[2] + 1e-2 * (t - t_sw) / grey_world.time_step_info['dt'],
                               tau_sw_params_final[2])
        grey_world.tau_sw_func_args = tuple(tau_sw_params)
    if tau_sw_params[2] == tau_sw_params_final[2]:
        # once sw optical depth reached max value, get to equilibrium
        data = grey_world.evolve_to_equilibrium(data, delta_net_flux_thresh, T_eqb.copy())
        # set sw optical depth to zero and then see how it evolves from there
        tau_sw_params[2] = 0
        grey_world.tau_sw_func_args = tuple(tau_sw_params)
        grey_world.update_grid()
        data = grey_world.evolve_to_equilibrium(data, delta_net_flux_thresh, T_eqb.copy())
        t = t_end + 10 # once reached final equilibrium, end simulation
    else:
        t = grey_world.update_temp(t, T_eqb.copy(), changing_tau)[0]
        data = grey_world.save_data(data, t)
anim = grey_world.plot_animate(data['T'], data['t'], T_eqb, correct_solution, data['tau'], data['flux'], nPlotFrames=30)
plt.show()

'''With stratosphere, evolve long wave to see if eventually get temp decrease'''
# tau_sw_params = [30000, 2000, 0.5]
# tau_lw_params_final = [100000, 8]
# tau_lw_params = [100000, 0.2]
# grey_world = Model.radiation.grey.GreyGas(nz='auto', tau_lw_func=OpticalDepthFunctions.exponential,
#                      tau_lw_func_args=tau_lw_params_final, tau_sw_func=OpticalDepthFunctions.peak_in_atmosphere,
#                      tau_sw_func_args=tau_sw_params)
# grey_world.tau_lw_func_args = tuple(tau_lw_params)
# grey_world.update_grid()
# up_flux_eqb, down_flux_eqb, T_eqb, up_sw_flux_eqb, down_sw_flux_eqb, \
#     correct_solution = grey_world.equilibrium_sol()
#
# t_array = [0]
# T_array = [grey_world.T.copy()]
# tau_array = {'lw': [grey_world.tau.copy()], 'sw': [grey_world.tau_sw.copy()]}
# flux_array = {'lw_up': [up_flux_eqb.copy()], 'lw_down': [down_flux_eqb.copy()],
#               'sw_up': [up_sw_flux_eqb.copy()], 'sw_down': [down_sw_flux_eqb.copy()]}
# data = {'t': t_array, 'T': T_array, 'tau': tau_array, 'flux': flux_array}
# changing_tau = True
# delta_net_flux_thresh = 1e-3
# keep_going = True
# while keep_going:
#     data = grey_world.evolve_to_equilibrium(data, delta_net_flux_thresh, T_eqb.copy())
#     tau_lw_params[1] = tau_lw_params[1] + 0.1
#     if tau_lw_params[1] > tau_lw_params_final[1]:
#         keep_going = False
#     else:
#         grey_world.tau_lw_func_args = tuple(tau_lw_params)
#         grey_world.update_grid()
# anim = grey_world.plot_animate(data['T'], data['t'], T_eqb, correct_solution, data['tau'], data['flux'],
#                                fract_frames_at_start=0, show_last_frame=True)
# plt.show()
# #anim.save('double_pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])