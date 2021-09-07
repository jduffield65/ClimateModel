from ..constants import g
import numpy as np
from . import numerical_methods
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable


class ShallowWater:
    def __init__(self, nx, ny, dx, dy, dt, f_0, beta, orography_info=None, initial_info=None,
                 boundary_type=None, numerical_solver='richtmyer'):
        """
        Sets horizontal velocities u and v as well as fluid depth, h, to initial values.

        :param nx: integer.
            Number of grid points in x direction.
        :param ny: integer.
            Number of grid points in y direction.
        :param dx: float.
            Length of grid square (m) in x direction
        :param dy: float.
            Length of grid square (m) in y direction.
        :param dt:  float.
            Time step (s).
        :param f_0: float.
            Grid is built on a beta plane, this is the value of the coriolis parameter (2 Omega sin(theta_0)
            where theta_0 is the latitude) at the middle of the plane.
        :param beta: float.
            Over the grid, the coriolis parameter, f = f_0 + beta x y. beta = df/dy at theta = theta_0.
            So beta = 2 Omega cos(theta_0) / R_planet where y = R_planet x theta.
        :param orography_info: dictionary, optional.
            This provides information to give to the function orography.
            default: None, meaning flat bottom surface.
        :param initial_info: dictionary, optional.
            This provides information to give to the function initial_conditions.
            default: None, meaning uniform westerley wind.
        :param boundary_type: dictionary, optional.
            Whether to have 'walls' or 'periodic' in the x and y directions.
            Periodic in 'y' doesn't make sense unless beta = 0.
            default: {'x': 'periodic', 'y': 'walls'}
        :param numerical_solver: string, optional.
            Name of numerical method used to solve non-linear equation of form
            dU/dt + dF(U)/dx + dG(U)/dy = Q(U).
            Options are 'lax_friedrichs', 'lax_wendroff', 'richtmyer', 'maccormack'.
            default: 'richtmyer'
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt_0 = dt
        self.dt = dt
        self.numerical_solver = numerical_solver
        self.numerical_func, self.numerical_args = self.get_numerical_func()
        self.orography_info = orography_info
        self.initial_info = initial_info
        if boundary_type is None:
            boundary_type = {'x': 'periodic', 'y': 'walls'}
        self.boundary_type = boundary_type
        x = np.mgrid[0:nx] * dx
        x = x - np.mean(x)  # Zonal distance coordinate (m), centre at 0
        y = np.mgrid[0:ny] * dy
        y = y - np.mean(y)  # Meridional distance coordinate (m), centre at 0
        [self.Y, self.X] = np.meshgrid(y, x)  # Create matrices of the coordinate variables
        self.f_0 = f_0
        self.f_coriolis = f_0 + beta * self.Y  # f0 = 2omega sin(theta). beta = 2omega cos(theta) / R_earth
        self.h_base = self.orography()
        self.u, self.v, self.h_surface = self.initial_conditions()
        self.h = self.h_surface - self.h_base  # fluid depth
        self.h, self.u, self.v = self.boundary_conditions(self.h, self.u, self.v)

    def get_numerical_func(self):
        """
        Returns the arguments required to call the desired numerical_function from numerical_methods.py
        """
        numerical_func = getattr(numerical_methods, self.numerical_solver)
        numerical_args = (self.flux_x, self.flux_y, self.source, self.dt, self.dx, self.dy, [0])
        if self.numerical_solver == 'lax_wendroff':
            numerical_args = numerical_args + (self.nx, self.ny, self.jacobian_x, self.jacobian_y)
        return numerical_func, numerical_args

    def orography(self):
        """
        Get profile of rigid base, h_base.
        Options are orography_info['type'] = 'flat', 'slope', or 'mountain'.
        Each option requires different information listed below.
        :return:
        h_base
        """
        if self.orography_info is None:
            self.orography_info = {'type': 'flat'}
        if self.orography_info['type'] == 'flat':
            # No bottom topography
            h_base = np.zeros((self.nx, self.ny))
        elif self.orography_info['type'] == 'slope':
            # Sloping floor in +x direction
            # floor reaches max height of orography_info['max_h_base'] at largest x value
            h_base = self.orography_info['max_h_base'] * (self.X - np.min(self.X)) / np.max(self.X)
        elif self.orography_info['type'] == 'mountain':
            # Mountain centered at (orography_info['x0'], orography_info['y0'])
            # with height orography_info['max_h_base'] and standard deviation
            # (orography_info['x_std'], orography_info['y_std'])
            h_base = self.orography_info['max_h_base'] * \
                     np.exp(-0.5 * ((self.X - self.orography_info['x0']) / self.orography_info['x_std']) ** 2 -
                            0.5 * ((self.Y - self.orography_info['y0']) / self.orography_info['y_std']) ** 2)
        else:
            raise ValueError("orography_info['type'] not valid")
        return h_base

    def get_geostrophic_velocities(self, h_surface):
        """
        Given an initial surface height distribution, this finds the
        corresponding velocities in geostrophic equilibrium.
        :param h_surface: numpy array [nx x ny].
        :return:
        u, v
        """
        u = np.zeros((self.nx, self.ny))
        v = np.zeros((self.nx, self.ny))
        u[1:-1, 1:-1] = -g * numerical_methods.centered_diff_y(h_surface, self.dy) / self.f_coriolis[1:-1, 1:-1]
        v[1:-1, 1:-1] = g * numerical_methods.centered_diff_x(h_surface, self.dx) / self.f_coriolis[1:-1, 1:-1]
        return u, v

    def initial_conditions(self):
        """
        Get starting conditions for u, v and h_surface.
        Options are initial_info['type'] = 'uniform_zonal', 'sinusoidal_zonal', 'jet_zonal',
        'height_gaussian', 'height_step'.
        All options require 'add_noise' which is 'False' or 'True'. Noise ma be added to trigger an instability.
        Otherwise, each option requires different information listed below.
        :return:
        u, v, h_surface
        """
        if self.initial_info is None:
            self.initial_info = {'type': 'uniform_zonal', 'mean_h_surface': 2 * np.max(self.h_base) + 1000,
                                 'u_mean': 20, 'add_noise': False}
        u = np.zeros((self.nx, self.ny))
        v = np.zeros((self.nx, self.ny))
        h_surface = np.ones((self.nx, self.ny))
        if self.initial_info['type'] == 'uniform_zonal':
            # zonal wind is the same everywhere
            # 'mean_h_surface' is height from which h_surface is adjusted to get wind of
            # desired speed 'u_mean' everywhere once geostrophic equilibrium has been applied.

            # compute surface so in geostrophic equilibrium
            h_surface = self.initial_info['mean_h_surface'] - (self.initial_info['u_mean'] * self.f_0 / g) * self.Y
            u, v = self.get_geostrophic_velocities(h_surface)
        elif self.initial_info['type'] == 'sinusoidal_zonal':
            # zonal wind oscillates between eastward to westward with latitude, y.
            # 'n_periods' is the number of full wavelengths of oscillation in the domain.
            # 'u_max' is the maximum zonal speed (only approximate if f_0 = 0).
            # 'y0' is where h_surface is a minimum.
            # 'mean_h_surface' is height from which h_surface oscillates
            cos_multiplier = 2 * self.initial_info['n_periods'] * np.pi / np.max(self.Y)
            if self.f_0 == 0:
                # Does not give make u_max equal to 'u_max' in this case but order of magnitude.
                h_jet_max = abs(self.f_coriolis).mean() * self.initial_info['u_max'] / (cos_multiplier * g)
            else:
                h_jet_max = self.f_0 * self.initial_info['u_max'] / (cos_multiplier * g)
            h_surface = self.initial_info['mean_h_surface'] - h_jet_max * \
                        np.cos((self.Y - self.initial_info['y0']) * cos_multiplier)
            u, v = self.get_geostrophic_velocities(h_surface)
        elif self.initial_info['type'] == 'jet_zonal':
            # sharp peak in zonal wind at a y = 'y0' of width 'jet_width' and max speed 'u_max'.
            # 'mean_h_surface' is height from which h_surface oscillates
            # Bickley jet is sech^2 so in geostrophic equilibrium, h is tanh.
            h_jet_max = self.f_0 * self.initial_info['u_max'] * self.initial_info['jet_width'] / g
            h_surface = self.initial_info['mean_h_surface'] - \
                        h_jet_max * np.tanh((self.Y - self.initial_info['y0']) / self.initial_info['jet_width'])
            u, v = self.get_geostrophic_velocities(h_surface)
        elif self.initial_info['type'] == 'height_gaussian':
            # gaussian peak in height of the surface at ('x0', 'y0') and standard deviation
            # ('x_std', 'y_std')
            # 'max_h_surface' is maximum height of gaussian blob (Can be below min_h_surface for negative gaussian).
            # 'min_h_surface' is the base height upon which the gaussian is added.
            h_surface = self.initial_info['min_h_surface'] + \
                        (self.initial_info['max_h_surface'] - self.initial_info['min_h_surface']) * \
                        np.exp(-0.5 * ((self.X - self.initial_info['x0']) / self.initial_info['x_std']) ** 2 -
                               0.5 * ((self.Y - self.initial_info['y0']) / self.initial_info['y_std']) ** 2)
        elif self.initial_info['type'] == 'height_step':
            # height discontinuosly increases from initial_info['min_h_surface'] to
            # initial_info['max_h_surface'] at initial_info['discontinuity_pos'].
            # 'direction' = 'x' or 'y' to indicate direction discontinuity is in.
            if self.initial_info['direction'] == 'y':
                min_zone = np.where(self.Y <= self.initial_info['discontinuity_pos'])
                max_zone = np.where(self.Y > self.initial_info['discontinuity_pos'])
            elif self.initial_info['direction'] == 'x':
                min_zone = np.where(self.X <= self.initial_info['discontinuity_pos'])
                max_zone = np.where(self.X > self.initial_info['discontinuity_pos'])
            h_surface[min_zone] = self.initial_info['min_h_surface']
            h_surface[max_zone] = self.initial_info['max_h_surface']
        else:
            raise ValueError("initial_info['type'] not valid")
        if self.initial_info['add_noise']:
            # Add noise that doesn't significantly change initial structure of h_surface.
            noise_amplitude = max(np.mean(abs(np.diff(h_surface)))/10, 1e-20)
            noise = np.random.randn(*np.shape(self.X)) * noise_amplitude
            h_surface = h_surface + noise
        if np.min(h_surface) < np.max(self.h_base):
            raise ValueError("surface height is less than floor height")
        return u, v, h_surface

    def update_time_step(self, target_courant=0.1):
        """
        For numerical stability, must have courant_number, sigma = max(|velocity|)*dt/dx < 1.
        This ensures courant number is below a target value.
        :param target_courant: float, optional.
            Courant number will be set to this as a maximum.
            default: 0.1
        """
        # keep courant number less than 1
        max_u = np.sqrt((self.u**2 + self.v**2).max())
        #max_u = np.max([abs(self.u).max(), abs(self.v).max()])
        self.dt = min(self.dt_0, target_courant * min(self.dx, self.dy) / max_u)
        if self.dt < 1e-20:
            raise ValueError("time step very small")
        numerical_args = list(self.numerical_args)
        numerical_args[3] = self.dt
        self.numerical_args = tuple(numerical_args)

    def time_step(self, t, data_dict=None):
        """
        This updates h, u and v by solving the shallow water equations for a single time step
        and then imposing the boundary conditions.
        :param t: float.
            Current time (s).
        :param data_dict: dictionary, optional.
            Dictionary containing lists of t, h, u and v data. If given, will append new values to it after
            time stepping.
            default: None.
        :return:
            t, data_dict
        """
        if data_dict is None:
            data_dict = {'t': [t], 'h': [self.h], 'u': [self.u], 'v': [self.v]}
        if t > 0:
            self.update_time_step()  # ensure courant number less than 1.
        U = self.get_conservative_form(self.h, self.u, self.v)
        U = self.numerical_func(U, *self.numerical_args)
        h, u, v = self.get_physical_values(U)
        self.h, self.u, self.v = self.boundary_conditions(h, u, v)
        if self.h.min() < 0:
            raise ValueError("surface height is less than floor height")
        t = t + self.dt
        data_dict = self.save_data(data_dict, t)
        return t, data_dict

    def save_data(self, data_dict, t):
        """
        This appends the time and current h, u, v to the data_dict.

        :param data_dict: dictionary.
            Must contain 't', 'h', 'u' and 'v' keys.
            The info in this dictionary is used to pass to plot_animate.
        :param t: float.
            Current time.
        :return:
            data_dict
        """
        data_dict['t'].append(t)
        data_dict['h'].append(self.h.copy())
        data_dict['u'].append(self.u.copy())
        data_dict['v'].append(self.v.copy())
        return data_dict

    def boundary_conditions(self, h, u, v):
        """
        Index 0 and -1 refer to ghost cells to deal with boundary conditions.
        At wall: normal velocity is zero and free slip so ghost cell value equals adjacent grid value
        i.e. field[0]=field[1] and field[-1]=field[-2].
        Periodic: To work out gradient correctly, require field[0,1] and field[-2,-1] to be the same.
        We know field[1] and field[-2] hence set field[0]=field[-2] and field[-1]=field[1].
        In this case, field[0,1] = field[-2,-1] = field[-2,1]

        :param h: numpy array [nx x ny].
            fluid depth
        :param u: numpy array [nx x ny].
            velocity in x direction.
        :param v: numpy array [nx x ny].
        :   velocity in y direction.
        :return:
            h, u, v
        """
        if self.boundary_type['x'] == 'periodic':
            for field in (h, u, v):
                # first x-slice
                field[0, 1:-1] = field[-2, 1:-1]
                field[0, 0] = field[-2, 1]
                field[0, -1] = field[-2, -2]
                # last x-slice
                field[-1, 1:-1] = field[1, 1:-1]
                field[-1, 0] = field[1, 1]
                field[-1, -1] = field[1, -2]
        elif self.boundary_type['x'] == 'walls':
            u[[0, -1], :] = 0
            for field in (h, v):
                field[0, :] = field[1, :]
                field[-1, :] = field[-2, :]

        if self.boundary_type['y'] == 'periodic':
            for field in (h, u, v):
                # first x-slice
                field[1:-1, 0] = field[1:-1, -2]
                field[0, 0] = field[1, -2]
                field[-1, 0] = field[-2, -2]
                # last x-slice
                field[1:-1, -1] = field[1:-1, 1]
                field[0, -1] = field[1, 1]
                field[-1, -1] = field[-2, -1]
        elif self.boundary_type['y'] == 'walls':
            # always wall in north/south
            v[:, [0, -1]] = 0.
            # no flux at y limits
            for field in (h, u):
                field[:, 0] = field[:, 1]
                field[:, -1] = field[:, -2]
        return h, u, v

    @staticmethod
    def get_conservative_form(h, u, v):
        """
        Put h, u, v into vector U = (h, uh, vh) for solving shallow water equations.

        :param h: numpy array [nx x ny].
            fluid depth
        :param u: numpy array [nx x ny].
            velocity in x direction.
        :param v: numpy array [nx x ny].
        :   velocity in y direction.
        :return:
        """
        U = np.zeros((3,) + np.shape(h))
        U[0] = h
        U[1] = h * u
        U[2] = h * v
        return U

    @staticmethod
    def get_physical_values(U):
        """
        Recover physical values from conserved values, U, after solving shallow water equations.

        :param U: numpy array [3 x nx x ny]
            U = (h, uh, vh)
        :return:
        """
        h = U[0]
        # set cases where divide by zero to 0 not nan.
        u = U[1] / h
        v = U[2] / h
        return h, u, v

    @staticmethod
    def flux_x(U):
        """
        returns F(U) in the equation dU/dt + dF(U)/dx + dG(U)/dy = Q(U)
        :param U: numpy array [3 x nx x ny]
            U = (h, uh, vh)
        """
        f = U.copy()
        f[0] = U[1]
        f[1] = U[1] ** 2 / U[0] + 0.5 * g * U[0] ** 2
        f[2] = U[1] * U[2] / U[0]
        return f

    @staticmethod
    def flux_y(U):
        """
        returns G(U) in the equation dU/dt + dF(U)/dx + dG(U)/dy = Q(U)
        :param U: numpy array [3 x nx x ny]
            U = (h, uh, vh)
        """
        f = U.copy()
        f[0] = U[2]
        f[1] = U[1] * U[2] / U[0]
        f[2] = U[2] ** 2 / U[0] + 0.5 * g * U[0] ** 2
        return f

    @staticmethod
    def jacobian_x(U):
        """
        The 'lax_wendroff' numerical method requires the jacobian matrix A = dF(U)/dU.
        :param U: numpy array [3 x nx x ny]
            U = (h, uh, vh)
        :return:
            A: [nx x ny x 3 x 3] numpy array.
        """
        A = np.zeros(np.shape(U[0, :, :]) + (3, 3))
        A[:, :, 1, 0] = -U[2] ** 2 / U[0] ** 2 + g * U[0]
        A[:, :, 2, 0] = -U[1] * U[2] / U[0] ** 2
        A[:, :, 0, 1] = 1
        A[:, :, 1, 1] = 2 * U[1] / U[0]
        A[:, :, 2, 1] = U[2] / U[0]
        A[:, :, 2, 2] = U[1] / U[0]
        return A

    @staticmethod
    def jacobian_y(U):
        """
        The 'lax_wendroff' numerical method requires the jacobian matrix B = dG(U)/dU.
        :param U: numpy array [3 x nx x ny]
            U = (h, uh, vh)
        :return:
            B: [nx x ny x 3 x 3] numpy array.
        """
        B = np.zeros(np.shape(U[0, :, :]) + (3, 3))
        B[:, :, 1, 0] = -U[1] * U[2] / U[0] ** 2
        B[:, :, 2, 0] = -U[2] ** 2 / U[0] ** 2 + g * U[0]
        B[:, :, 1, 1] = U[2] / U[0]
        B[:, :, 0, 2] = 1
        B[:, :, 1, 2] = U[1] / U[0]
        B[:, :, 2, 2] = 2 * U[2] / U[0]
        return B

    def source(self, U):
        """
        returns Q(U) in the equation dU/dt + dF(U)/dx + dG(U)/dy = Q(U)
        As Q[0] = 0, can solve for U[0] first and then use updated U[0] to solve for U[1] and U[2].
        :param U: numpy array [3 x nx x ny]
            U = (h, uh, vh)
        """
        Q = np.zeros(np.subtract(np.shape(U), (0, 2, 2)))
        h, u, v = self.get_physical_values(U[:, 1:-1, 1:-1])
        Q[1] = h * (self.f_coriolis[1:-1, 1:-1] * v -
                    g * numerical_methods.centered_diff_x(self.h_base, self.dx))
        Q[2] = h * (-self.f_coriolis[1:-1, 1:-1] * u -
                    g * numerical_methods.centered_diff_y(self.h_base, self.dy))
        return Q

    def plot_animate(self, t_array, h_array, u_array, v_array, nPlotFrames=50, fract_frames_at_start=0.0):
        """
        This plots an animation showing the evolution of the surface height and vorticity with time.
        On the height plot, arrows are added showing the velocity.
        If the base is not flat, contours of the base will be plotted.

        :param t_array: list.
            times where information collected in simulation.
        :param h_array: list of numpy arrays.
            h_array[i] is the [nx x ny] fluid depth field at t = t_array[i]
        :param u_array: list of numpy arrays.
            u_array[i] is the [nx x ny] x velocity field at t = t_array[i]
        :param v_array: list of numpy arrays.
            v_array[i] is the [nx x ny] y velocity field at t = t_array[i]
        :param nPlotFrames: integer, optional.
            Not all values in the above arrays will be plotted, only nPlotFrames spanning the
            whole time range will.
            default: 50
        :param fract_frames_at_start: float, optional.
            fract_frames_at_start*nPlotFrames of the animation frames will be at the start.
            The remainder will be equally spaced amongst the remaining times.
            default: 0.0
        """
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
        # separate axis for colorbars so can clear them.
        div = make_axes_locatable(axs[0])
        cax1 = div.append_axes('right', '5%', '5%')
        div = make_axes_locatable(axs[1])
        cax2 = div.append_axes('right', '5%', '5%')
        interval = int(min([6, self.ny / 5, self.nx / 5]))

        t_plot = np.array(t_array)
        h_plot = np.array(h_array)
        u_plot = np.array(u_array)
        v_plot = np.array(v_array)
        if np.size(t_plot) > nPlotFrames:
            start_end_ind = int(fract_frames_at_start * nPlotFrames)
            use_plot_start = np.arange(0, start_end_ind)
            use_plot_end = np.unique(np.linspace(start_end_ind, np.size(t_plot) - 1,
                                                 int((1 - fract_frames_at_start) * nPlotFrames), dtype='int'))[1:]
            use_plot = np.concatenate((use_plot_start, use_plot_end))
            t_plot = t_plot[use_plot]
            h_plot = h_plot[use_plot]
            u_plot = u_plot[use_plot]
            v_plot = v_plot[use_plot]

        x = self.X[1:-1, 0]
        y = self.Y[0, 1:-1]
        # plot surface height not fluid depth
        h_base = self.h_base[1:-1, 1:-1]
        h_surface_plot = h_plot[:, 1:-1, 1:-1] + h_base
        # diverging colormap eitherside of median initial surface height.
        median_h_surface = np.median(self.h_surface)
        h_diff_from_median_max = abs(h_surface_plot - median_h_surface).max()
        h_caxis_lims = tuple([median_h_surface - h_diff_from_median_max,
                              median_h_surface + h_diff_from_median_max])
        # diverging colormap eitherside of 0
        min_grid_space = min([self.dx, self.dy])
        velocity_max = max([abs(u_plot).max(), abs(v_plot).max()])
        vort_max = max([abs(u_plot).max() / self.dy, abs(v_plot).max() / self.dx])
        vort_caxis_lims = (-velocity_max / min_grid_space, velocity_max / min_grid_space)
        velocity_scale = min_grid_space * interval / velocity_max  # so arrows show up
        h_base_max = self.h_base.max()
        h_contour_interval = h_base_max / 5

        def animate(i, shallow_world):
            '''What to do at each frame of animation'''
            cax1.cla()
            cax2.cla()
            axs[0].clear()
            axs[1].clear()
            h_surface = h_surface_plot[i]
            u = u_plot[i][1:-1, 1:-1]
            v = v_plot[i][1:-1, 1:-1]

            vorticity = numerical_methods.centered_diff_x(v_plot[i], shallow_world.dx) - \
                        numerical_methods.centered_diff_y(u_plot[i], shallow_world.dy)

            # Plot the height field
            im = axs[0].imshow(np.transpose(h_surface),
                               extent=[np.min(x), np.max(x), np.min(y), np.max(y)], cmap='bwr', origin='lower')
            # Set other axes properties and plot a colorbar
            cb1 = fig.colorbar(im, cax=cax1)
            cb1.set_label('height (m)')
            # Contour the terrain:
            if shallow_world.orography_info['type'] != 'flat':
                cs = axs[0].contour(x, y, np.transpose(h_base),
                                    levels=range(1, h_base_max, h_contour_interval), colors='k')
            # Plot the velocity vectors
            f = lambda x: x * velocity_scale
            Q = axs[0].quiver(x[2::interval], y[2::interval],
                              f(np.transpose(u[2::interval, 2::interval])),
                              f(np.transpose(v[2::interval, 2::interval])),
                              scale_units='xy', scale=1, minshaft=2, pivot='mid')
            axs[0].set_ylabel('Y distance (m)')

            # Now plot the vorticity
            im2 = axs[1].imshow(np.transpose(vorticity), extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
                                cmap='bwr', origin='lower')
            # Set other axes properties and plot a colorbar
            cb2 = fig.colorbar(im2, cax=cax2)
            cb2.set_label('vorticity (s$^{-1}$)')
            axs[1].set_xlabel('X distance (m)')
            axs[1].set_ylabel('Y distance (m)')
            # axs[1].set_title('Relative vorticity (s$^{-1}$)')
            # tx2 = ax2.text(0, np.max(y_1000km), 'Time = %.1f hours' % (t_save[it] / 3600.))

            im.set_clim(h_caxis_lims)
            im2.set_clim(vort_caxis_lims)
            axs[0].axis((np.min(x), np.max(x), np.min(y), np.max(y)))
            axs[1].axis((np.min(x), np.max(x), np.min(y), np.max(y)))
            t_full_hours = t_plot[i] / 60 ** 2
            t_days, t_hours = divmod(t_full_hours, 24)
            Title = "{:.0f}".format(t_days) + " Days and " + "{:.1f}".format(t_hours) + " Hours"
            axs[0].text(0.5, 1.01, Title, horizontalalignment='center', verticalalignment='bottom',
                        transform=axs[0].transAxes)

        # animate(0, self)
        anim = FuncAnimation(fig, animate, frames=np.size(t_plot), interval=100,
                             blit=False, repeat_delay=200, fargs=(self,))
        return anim
