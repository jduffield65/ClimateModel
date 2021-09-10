import matplotlib.pyplot as plt
from Model.dynamics.shallow_water import ShallowWater
from Model.constants import g, R_earth, Omega
import numpy as np
import matplotlib
from tqdm import tqdm

matplotlib.use('TkAgg')  # To make plot pop out

"""Default values"""
n_days = 4
nx = 254  # Number of zonal gridpoints
ny = 50  # Number of meridional gridpoints
dt = 60  # s
dx = 100.0e3  # Zonal grid spacing (m)
dy = dx  # Meridional grid spacing
min_h_surface = 9750
max_h_surface = min_h_surface + 1000
f_0 = 1e-4  # Coriolis parameter (s-1)
beta = 1.6e-11  # Meridional gradient of f (s-1m-1)
boundary_type = {'x': 'periodic', 'y': 'walls'}
orography_info = None
r = 0  # damping
g_use = g
linear = False
save_every = 0.1
L_def = None

"""Geostrophic adjustment"""
# deform_radius = dx * 1
# min_h_surface = (f_0*deform_radius)**2/g
# max_h_surface = min_h_surface * 1.2
# initial_info = {'type': 'height_step', 'direction': 'x', 'discontinuity_pos': 0,
#                 'min_h_surface': min_h_surface, 'max_h_surface': max_h_surface, 'add_noise': False}
# beta = 0
# boundary_type = {'x': 'walls', 'y': 'periodic'}

"""Gravity wave"""
# n_days = 1.5
# ny = nx
# f_0 = 0
# beta = 0
# h_surface_blob_std = 8*dy
# initial_info = {'type': 'height_gaussian', 'min_h_surface': min_h_surface, 'max_h_surface': max_h_surface,
#                 'x0': -9487500, 'y0': 0, 'x_std': h_surface_blob_std, 'y_std': h_surface_blob_std, 'add_noise': False}

"""Tsunami"""
# n_days = 1.5
# ny = nx
# f_0 = 0
# beta = 0
# h_surface_blob_std = 8*dy
# initial_info = {'type': 'height_gaussian', 'min_h_surface': min_h_surface, 'max_h_surface': max_h_surface,
#                 'x0': -9487500, 'y0': 0, 'x_std': h_surface_blob_std, 'y_std': h_surface_blob_std, 'add_noise': False}
# mount_std = 40*dy
# orography_info = {'type': 'mountain', 'max_h_base': 9250,
#                   'x0': 0, 'y0': -12*dy, 'x_std': mount_std, 'y_std': mount_std}

"""Barotropic Instability"""
# initial_info = {'type': 'jet_zonal', 'u_max': 400, 'jet_width': dy,
#                 'mean_h_surface': min_h_surface, 'y0': 0, 'add_noise': True}

"""Jupiter Red Spot"""
# n_days = 10
# initial_info = {'type': 'sinusoidal_zonal', 'u_max': 100, 'n_periods': 1,
#                 'mean_h_surface': min_h_surface, 'y0': 0, 'add_noise': True}

"""Rossby Mountain Waves"""
# n_days = 10
# #beta = 0
# initial_info = {'type': 'uniform_zonal', 'mean_h_surface': 1000, 'u_mean': 10, 'add_noise': False}
# mount_std = 5*dy
# orography_info = {'type': 'mountain', 'max_h_base': 500,
#                   'x0': 0, 'y0': 0, 'x_std': mount_std, 'y_std': mount_std}

"""Equatorally trapped waves"""
# n_days = 10
# f_0 = 0
# beta = 2.5e-10
# initial_info = {'type': 'sinusoidal_zonal', 'u_max': 90, 'n_periods': 1,
#                 'mean_h_surface': min_h_surface, 'y0': 0, 'add_noise': True}

"""Equatorial Kelvin Wave"""
# ny = 100
# n_days = 1
# f_0 = 0
# beta = 5e-10
# h_surface_blob_std = 8*dy
# initial_info = {'type': 'height_gaussian', 'min_h_surface': min_h_surface, 'max_h_surface': max_h_surface,
#                 'x0': 0, 'y0': 0, 'x_std': h_surface_blob_std, 'y_std': h_surface_blob_std, 'add_noise': False}
# boundary_type = {'x': 'walls', 'y': 'walls'}

"""El Nino"""
f_0 = 0
h_mean = 100
g_use = 0.05
c = np.sqrt(g_use * h_mean)
beta = 2 * Omega / R_earth
L_def = np.sqrt(c / beta)
t_def = L_def / c
dx = L_def / 5
dy = dx
nx = int(round(30 * L_def / dx))
ny = int(round(15 * L_def / dy))
courant_target = 0.01
dt = courant_target * dx / c
r = 1 / (10 * 30 * 24 * 60 ** 2)  # damping time scale of 16 months
n_days = 1600.0
save_every = 24*60**2
y_walls_damp = {'dist_thresh': (ny/2)*dy-6*dy, 'r': r * 100}
boundary_type = {'x': 'walls', 'y': 'walls', 'y_walls_damp': y_walls_damp}
h_perturb = h_mean / 10
linear = False
wind_dict = {'type': 'seasonal_forced'}
initial_info = {'type': 'el_nino', 'max_h_surface': h_mean+h_perturb, 'min_h_surface': h_mean-h_perturb,
                'y_std': L_def, 'add_noise': False, 'wind': wind_dict}


"""Run simulation"""
shallow_world = ShallowWater(nx, ny, dx, dy, dt, f_0, beta, initial_info=initial_info,
                             numerical_solver='richtmyer', boundary_type=boundary_type,
                             orography_info=orography_info, r=r, g=g_use, linear=linear)

forecast_length = n_days * 24.0 * 60 ** 2  # Forecast length (s)
nt = int(np.fix(forecast_length / dt) + 1)  # Number of timesteps
t = 0
data_dict = {'t': [t], 'h': [shallow_world.h], 'u': [shallow_world.u], 'v': [shallow_world.v]}
for n in tqdm(range(0, nt)):
    t, data_dict = shallow_world.time_step(t, data_dict, save_every=save_every)

if initial_info['type'] == 'el_nino':
    fig = shallow_world.el_nino_plot(np.array(data_dict['t']), np.array(data_dict['h']))
anim = shallow_world.plot_animate(data_dict['t'], data_dict['h'], data_dict['u'], data_dict['v'],
                                  nPlotFrames=50, fract_frames_at_start=0)
# anim.save('shallow.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
