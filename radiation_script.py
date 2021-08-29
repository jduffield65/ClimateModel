import matplotlib.pyplot as plt
import Model.radiation.grey
import matplotlib
import numpy as np
from Model.constants import p_surface
from matplotlib.animation import FuncAnimation
from Model.radiation.grey import OpticalDepthFunctions
from Model.constants import p_surface

matplotlib.use('TkAgg')  # To make plot pop out

'''Analytic solution with short wave'''
# p_width_lw = 100000
# # alpha ratio must be integer for analytic solution
# alpha_sw = OpticalDepthFunctions.get_exponential_alpha(p_width_lw) / 5
# p_width_sw = OpticalDepthFunctions.get_exponential_p_width(alpha_sw)
# grey_world = Model.radiation.grey.GreyGas(nz='auto', tau_lw_func=OpticalDepthFunctions.exponential,
#                                           tau_lw_func_args=[p_width_lw, 4], tau_sw_func=OpticalDepthFunctions.exponential,
#                                           tau_sw_func_args=[p_width_sw, 0.6])
'''With stratosphere'''
# grey_world = Model.radiation.grey.GreyGas(nz='auto', tau_lw_func=OpticalDepthFunctions.exponential,
#                                           tau_lw_func_args=[100000, 4], tau_sw_func=OpticalDepthFunctions.peak_in_atmosphere,
#                                           tau_sw_func_args=[30000, 2000, 0.5])
'''With mesosphere'''
# grey_world = Model.radiation.grey.GreyGas(nz='auto', tau_lw_func=OpticalDepthFunctions.scale_height_and_peak_in_atmosphere,
#                                           tau_lw_func_args=[50000, 4, 1000, 600, 0.3], tau_sw_func=OpticalDepthFunctions.peak_in_atmosphere,
#                                           tau_sw_func_args=[10000, 2000, 0.05])
'''With thermosphere'''
# grey_world = Model.radiation.grey.GreyGas(nz='auto', tau_lw_func=OpticalDepthFunctions.scale_height_and_peak_in_atmosphere,
#                                           tau_lw_func_args=[51000, 4, 100, 600, 0.1], tau_sw_func=OpticalDepthFunctions.scale_height_and_peak_in_atmosphere,
#                                           tau_sw_func_args=[p_surface, 0.12, 100, 20, 0.002])

""" Approach to equilibrium"""
# up_flux_eqb, down_flux_eqb, T_eqb, up_sw_flux_eqb, down_sw_flux_eqb, \
#     correct_solution = grey_world.equilibrium_sol()
# if correct_solution:
#     grey_world.plot_eqb(up_flux_eqb, down_flux_eqb, T_eqb, up_sw_flux_eqb, down_sw_flux_eqb)
# #plt.show()
# t = 0
# # Get temperature results until net_flux is zero everywhere i.e. equilibrium
# net_flux_thresh = 1e-1
# func_net_flux_thresh = 1e-7
# t_array = [0]
# T_array = [grey_world.T.copy()]
# n_no_flux_change = 0
# no_flux_change_thresh = 100
# while (max(abs(grey_world.net_flux)) > net_flux_thresh or t == 0) and n_no_flux_change < no_flux_change_thresh:
#     if t > 365*24*60**2:
#         func_net_flux_thresh = 0.1
#     t, delta_net_flux = grey_world.update_temp(t)
#     n_no_flux_change = n_no_flux_change + int(delta_net_flux < net_flux_thresh)
#     t_array.append(t)
#     T_array.append(grey_world.T.copy())
#     if min(grey_world.T)[0] < 0:
#         raise ValueError('Temperature is below zero')
#
# anim = grey_world.plot_animate(T_array, t_array, T_eqb, correct_solution)
# plt.show()


""" Evolution with tau"""
p1 = OpticalDepthFunctions.get_exponential_p_width(1e-5)
p2 = OpticalDepthFunctions.get_exponential_p_width(1e-5 / 3, 2000)
tau_params_final = [100000, 6]
tau_params = [100000, 4]
p_max = 2000
tau_sw_params_final = [300000, 2000, 1.2]
tau_sw_params = [300000, 2000, 0]
grey_world = Model.radiation.grey.GreyGas(nz='auto', tau_lw_func=OpticalDepthFunctions.exponential,
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
t_turn_off = t_end
t_array = [0]
T_array = [T_eqb.copy()]
tau_array = {'lw': [grey_world.tau.copy()], 'sw': [grey_world.tau_sw.copy()]}
flux_array = {'lw_up': [up_flux_eqb.copy()], 'lw_down': [down_flux_eqb.copy()],
              'sw_up': [up_sw_flux_eqb.copy()], 'sw_down': [down_sw_flux_eqb.copy()]}
# tau_array = [grey_world.tau.copy()]
# tau_sw_array = [grey_world.tau_sw.copy()]
changing_tau = True
delta_net_flux_thresh = 1e-4
while t < t_end:
    tau_params[1] = min(tau_params[1] + 1e-8 * t, tau_params_final[1])
    grey_world.tau_lw_func_args = tuple(tau_params)
    if tau_params[1] == tau_params_final[1]:
        if t_sw == t_end:
            changing_tau = False
            if delta_net_flux < delta_net_flux_thresh:
                changing_tau = True
        if changing_tau:
            grey_world.time_step_info['RemoveInd'] = []
            t_sw = min(t, t_sw)
            tau_sw_params[2] = min(tau_sw_params[2] + 1e-2 * (t - t_sw) / grey_world.time_step_info['dt'],
                                   tau_sw_params_final[2])
            grey_world.tau_sw_func_args = tuple(tau_sw_params)
    if tau_sw_params[2] == tau_sw_params_final[2]:
        changing_tau = False
        if delta_net_flux < delta_net_flux_thresh:
            changing_tau = True
            grey_world.time_step_info['RemoveInd'] = []
            tau_sw_params[2] = 0
            grey_world.tau_sw_func_args = tuple(tau_sw_params)
            t_turn_off = t
    t, delta_net_flux = grey_world.update_temp(t, T_eqb.copy(), changing_tau)
    if t_turn_off < t_end:
        changing_tau = False
    t_array.append(t)
    T_array.append(grey_world.T.copy())
    tau_array['lw'].append(grey_world.tau.copy())
    tau_array['sw'].append(grey_world.tau_sw.copy())
    flux_array['lw_up'].append(grey_world.up_lw_flux.copy())
    flux_array['lw_down'].append(grey_world.down_lw_flux.copy())
    flux_array['sw_up'].append(grey_world.up_sw_flux.copy())
    flux_array['sw_down'].append(grey_world.down_sw_flux.copy())
    if min(grey_world.T)[0] < 0:
        raise ValueError('Temperature is below zero')

anim = grey_world.plot_animate(T_array, t_array, T_eqb, correct_solution, tau_array, flux_array)
plt.show()
