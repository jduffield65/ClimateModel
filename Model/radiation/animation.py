from .base import t_years_days
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation


class Animate:

    flux_plot_max_ax_lim = 5.0  # on flux plot, if possible have axes min and max less than this
    lw_color = '#ff7f0e' # color of long wave plot info
    sw_color = '#1f77b4' # color of short wave plot info

    def __init__(self, atmos, T_array, t_array, T_eqb=None, correct_solution=True, tau_array=None, flux_array=None,
                 q_array=None, log_axis=True, nPlotFrames=100, fract_frames_at_start=0.25, start_step=3,
                 show_last_frame=False):
        """
        This plots an animation showing the evolution of the temperature profile, optical depth profile and fluxes
        with time.

        :param atmos: Atmosphere class object
        :param T_array: list of numpy arrays.
            T_array[i] is the temperature profile at time t_array[i]
        :param t_array: list.
            times where temperature profile was updated in the simulation.
        :param T_eqb: numpy array, optional.
            If given, analytic equilibrium solution will also be plotted.
            default: None.
        :param correct_solution: boolean, optional.
            If T_eqb given, this indicates whether the analytic solution was correct. Affects label.
        :param tau_array: dictionary, optional.
            tau_array['lw'][i] is long wave optical depth profile at time t_array[i].
            tau_array['sw'][i] is short wave optical depth profile at time t_array[i].
            If given, will add second subplot showing evolution of long wave optical depth.
            Only really makes sense to plot if optical depth was changing with time.
            default: None.
        :param flux_array: dictionary, optional.
            flux_array['lw_up'][i] is long wave upward flux at time t_array[i].
            flux_array['lw_down'][i] is long wave downward flux at time t_array[i].
            flux_array['sw_up'][i] is short wave upward flux at time t_array[i].
            flux_array['sw_down'][i] is short wave downward flux at time t_array[i].
            If given will add subplot showing evolution of flux with time.
            default: None.
        :param q_array: dictionary, optional.
            q_array[molecule_name][i] is the concentration distribution in ppmv of that molecule
            at the time t_array[i].
        :param log_axis: boolean, optional.
            Whether to have pressure on the y-axis as a log-scale.
            default: True.
        :param nPlotFrames: integer, optional.
            Number of frames to show in the animation.
            default: 100.
        :param fract_frames_at_start: float, optional.
            fract_frames_at_start*nPlotFrames of the animation frames will be at the start.
            The remainder will be equally spaced amongst the remaining times.
            default: 0.25
        :param start_step: integer, optional.
            The step size of the first fract_frames_at_start*nPlotFrames frames.
            A value of 2 would mean every other time in the first fract_frames_at_start*nPlotFrames*2 frames.
            default: 3
        :param show_last_frame:  boolean, optional.
            If True, will plot the last frame collected. Otherwise will only plot till temperature stops changing.
            default: False
        """
        if atmos.ny > 1:
            # 2D plot
            self.plot_type = 2
        else:
            self.plot_type = 1
        # assign all variables to class with same name as in __init__ def
        for name, value in vars().items():
            if name != 'self':
                setattr(self, name, value)
        if str(atmos) == 'Real Gas':
            # don't know equilibrium for real gas
            self.T_eqb = None
        self.t_plot = None
        self.T_plot = None
        self.compos_plot = None  # q or if q not given, tau. Both indicate composition of atmosphere.
        self.flux_plot = None
        self.get_plot_data()
        self.plot2_info = {}
        fig, ax, axs = self.plot_setup()
        self.ax_lims = {}
        self.get_axs_lims()
        self.plot_labels = {}
        self.get_plot_labels()
        if self.T_eqb is None:
            # if don't know equilibrium result, just use final value.
            self.T_eqb = self.T_array[-1]
        if self.plot_type == 2:
            self.update_plot2_info()
            self.anim = FuncAnimation(fig, self.animate_func_2d,
                                      frames=np.size(self.t_plot), interval=100, blit=False, repeat_delay=2000)
        else:
            self.anim = FuncAnimation(fig, self.animate_func_1d,
                                      frames=np.size(self.t_plot), interval=100, blit=False, repeat_delay=2000,
                                      fargs=(ax, axs))

    def get_plot_data(self):
        """
        Get subsection of data for plotting
        """
        if self.plot_type == 2 and self.tau_array is not None:
            for key in [*self.tau_array]:
                # assume tau is same at all latitudes so choose first latitude.
                self.tau_array[key] = np.array(self.tau_array[key])[:, :, 0]
        F_norm = self.atmos.F_stellar_constant / 4  # normalisation for flux plots
        if len(self.T_array) > self.nPlotFrames:
            start_end_ind = self.start_step * int(self.fract_frames_at_start * self.nPlotFrames)
            use_plot_start = np.arange(0, start_end_ind, self.start_step)
            # find max_index beyond which temperature is constant
            index_where_temp_diff_small = np.where(np.percentile(np.abs(np.diff(self.T_array, axis=0)),
                                                                 99, axis=1) < 0.01)[0]
            index_sep = np.where(np.ediff1d(index_where_temp_diff_small) > 1)[0]
            if len(index_sep) == 0:
                if len(index_where_temp_diff_small) == 0:
                    max_index = len(self.T_array) - 1
                else:
                    max_index = index_where_temp_diff_small[0] + 1
            else:
                max_index = index_where_temp_diff_small[max(index_sep) + 1] + 1
            if self.show_last_frame:
                max_index = len(self.T_array) - 1
            use_plot_end = np.linspace(start_end_ind, max_index,
                                       int((1 - self.fract_frames_at_start) * self.nPlotFrames), dtype=int)
            use_plot = np.unique(np.concatenate((use_plot_start, use_plot_end)))
        else:
            use_plot = np.arange(len(self.T_array))
        self.T_plot = np.array(self.T_array)[use_plot]
        self.t_plot = np.array(self.t_array)[use_plot]
        if self.flux_array is not None and self.plot_type == 1:
            self.flux_plot = {}
            for key in self.flux_array.keys():
                self.flux_plot[key] = np.array(self.flux_array[key])[use_plot] / F_norm
            self.flux_plot['net'] = self.flux_plot['lw_up'] + self.flux_plot['sw_up'] - \
                                    self.flux_plot['lw_down'] - self.flux_plot['sw_down']
        # only have one composition plot, q takes priority.
        if self.q_array is not None:
            self.compos_plot = {}
            for key in self.q_array.keys():
                self.compos_plot[key] = np.array(self.q_array[key])[use_plot]
        elif self.tau_array is not None:
            self.compos_plot = {'short wave': np.array(self.tau_array['sw'])[use_plot],
                                'long wave': np.array(self.tau_array['lw'])[use_plot]}

    def plot_setup(self):
        """
        returns ax (1st subplot of plot_type 1) and axs (all axes in subplot)
        Also adds info to self.plot2_info if 2D plot
        """
        if self.plot_type == 2:
            if self.compos_plot is not None:
                fig, axs = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
                gs = axs[1, 1].get_gridspec()
                for ax in axs[-1, :]:
                    ax.remove()
                self.plot2_info['axTemp'] = fig.add_subplot(gs[-1, :])
                axs[0, 1].get_shared_y_axes().join(axs[0, 1], axs[0, 0])
                axs[0, 1].get_shared_x_axes().join(axs[0, 1], self.plot2_info['axTemp'])
                self.plot2_info['axColor'] = axs[0, 1]
                self.plot2_info['ax_compos'] = axs[0, 0]
            else:
                fig, (self.plot2_info['axColor'], self.plot2_info['axTemp']) = \
                    plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={'height_ratios': [3, 1]})
                axs = None
            self.plot2_info['fig'] = fig
            ax = None
        else:
            nPlots = 1 + int(self.compos_plot is not None) + int(self.flux_plot is not None)
            if nPlots > 1:
                fig, axs = plt.subplots(1, nPlots, sharey=True, figsize=(6 * nPlots, 5))
                ax = axs[0]
            else:
                fig, ax = plt.subplots(1, 1)
                axs = None
        return fig, ax, axs

    def get_axs_lims(self):
        """
        Updates ax_lims dictionary to include axes limits.
        """
        T_min = min([min(self.T_plot[i].flatten()) for i in range(len(self.T_plot))]) - 10
        T_max = max([max(self.T_plot[i].flatten()) for i in range(len(self.T_plot))]) + 10
        if self.T_eqb is not None:
            T_min = min([min(self.T_eqb.flatten()) - 10, T_min])
            T_max = max([max(self.T_eqb.flatten()) + 10, T_max])
        self.ax_lims['T'] = (T_min,T_max)
        self.ax_lims['p'] = (self.atmos.p_toa, self.atmos.p_surface)
        if self.compos_plot is not None:
            if self.q_array is None:
                compos_min = -0.1
            else:
                # log axis if using q hence can't be zero min axes
                compos_min = min(min(i for v in self.compos_plot[key] for i in v[v > 0])
                                 for key in [*self.compos_plot])
            compos_max = max(max(i for v in self.compos_plot[key] for i in v)
                             for key in [*self.compos_plot]) + 1
            self.ax_lims['compos'] = (compos_min, compos_max)

        if self.flux_plot is not None:
            flux_min = -max(max(i for v in self.flux_plot[key] for i in v) for key in ['lw_down', 'sw_down']) - 0.1
            flux_max = max(max(i for v in self.flux_plot[key] for i in v) for key in ['lw_up', 'sw_up']) + 0.1
            self.ax_lims['flux'] = [flux_min, flux_max]

    def get_plot_labels(self):
        if self.T_eqb is not None:
            if self.correct_solution and not self.atmos.sw_tau_is_zero:
                T_eqb_label = r'Radiative Equilibrium, $\tau_{sw}\neq 0$'
                Tcurrent_label = r'Current, $\tau_{sw}\neq0$'
            elif self.correct_solution:
                T_eqb_label = r'Radiative Equilibrium, $\tau_{sw}=0$'
                Tcurrent_label = r'Current, $\tau_{sw}=0$'
            else:
                T_eqb_label = r'Radiative Equilibrium, $\tau_{sw}=0$ (Wrong)'
                Tcurrent_label = r'Current, $\tau_{sw}\neq0$'
        else:
            T_eqb_label = 'Final'
            Tcurrent_label = 'Current'
        if self.tau_array is not None:
            Tcurrent_label = 'Current'
        self.plot_labels['T_current'] = Tcurrent_label
        self.plot_labels['T_eqb'] = T_eqb_label
        if self.flux_plot is not None:
            self.plot_labels['flux'] = {'initial': {}, 'current': {}, 'color': {}, 'sign': {}}
            self.plot_labels['flux']['initial'] = {'sw_up': '$F_{sw}(t=0)$', 'sw_down': None,
                                                   'lw_up': '$F_{lw}(t=0)$', 'lw_down': None}
            self.plot_labels['flux']['current'] = {'sw_up': '$F_{sw}$', 'sw_down': None,
                                                   'lw_up': '$F_{lw}$', 'lw_down': None}
            self.plot_labels['flux']['color'] = {'sw_up': Animate.sw_color, 'sw_down': Animate.sw_color,
                                                 'lw_up': Animate.lw_color, 'lw_down': Animate.lw_color}
            # down shows up as negative
            self.plot_labels['flux']['sign'] = {'sw_up': 1.0, 'sw_down': -1.0,
                                                'lw_up': 1.0, 'lw_down': -1.0}


    def update_plot2_info(self):
        div = make_axes_locatable(self.plot2_info['axColor'])
        self.plot2_info['cax'] = div.append_axes('right', '5%', '5%')
        n_labels_keep = min([len(self.atmos.latitude), 6])
        keep_x_ticks = np.linspace(0, len(self.atmos.latitude) - 1, n_labels_keep, dtype='int')
        self.plot2_info['miss_x_ticks'] = np.setdiff1d(np.linspace(0, len(self.atmos.latitude) - 1,
                                                                   len(self.atmos.latitude), dtype='int'), keep_x_ticks)
        self.plot2_info['X'], self.plot2_info['Y'] = np.meshgrid(np.array(self.atmos.latitude), self.atmos.p[:, 0])

    def animate_func_1d(self, i, ax, axs):
        """
        function to pass to FuncAnimation
        i.e. what to do at each frame of the animation
        :param i: current frame index
        :param self: Animation object
        :param ax: axes of temperature plot
        :param axs: numpy array containing all axes in plot
        """
        ax.clear()
        ax.plot(self.T_plot[0], self.atmos.p,
                label='Initial', color=Animate.sw_color, linestyle='dotted')
        ax.plot(self.T_eqb, self.atmos.p, label=self.plot_labels['T_eqb'], color=Animate.lw_color, linestyle='dotted')
        ax.plot(self.T_plot[i], self.atmos.p, label=self.plot_labels['T_current'], color='#d62728')
        ax.set_ylim(self.ax_lims['p'])
        if self.log_axis:
            ax.set_yscale('log')
        ax.axes.invert_yaxis()
        ax.set_xlabel('Temperature / K')
        ax.set_ylabel('Pressure / Pa')
        ax.set_xlim(self.ax_lims['T'])
        ax.legend()
        if self.compos_plot is not None:
            axs[1].clear()
            for key in self.compos_plot:
                axs[1].plot(self.compos_plot[key][0], self.atmos.p, linestyle='dotted')  # initial values
                axs[1].plot(self.compos_plot[key][i], self.atmos.p, label=key, color=axs[1].lines[-1].get_color())
            if self.q_array is None:
                axs[1].set_xlabel(r'$\tau$')
            else:
                # log scale if plotting q
                axs[1].set_xlabel('Volume Mixing Ration (ppmv)')
                axs[1].set_xscale('log')
            axs[1].set_xlim(self.ax_lims['compos'])
            if self.log_axis:
                axs[1].set_yscale('log')
            axs[1].legend()
        if self.flux_plot is not None:
            axs[-1].clear()
            for key in list(self.flux_plot.keys())[:-1]:
                axs[-1].plot(self.flux_plot[key][0] * self.plot_labels['flux']['sign'][key], self.atmos.p_interface,
                             color=self.plot_labels['flux']['color'][key], linestyle='dotted',
                             label=self.plot_labels['flux']['initial'][key])
            for key in list(self.flux_plot.keys())[:-1]:
                axs[-1].plot(self.flux_plot[key][i] * self.plot_labels['flux']['sign'][key], self.atmos.p_interface,
                             color=self.plot_labels['flux']['color'][key],
                             label=self.plot_labels['flux']['current'][key])
            axs[-1].plot(self.flux_plot['net'][i], self.atmos.p_interface, label='$F_{net}$', color='#d62728')
            axs[-1].set_xlabel(r'Radiation Flux, $F$, as fraction of Incoming Solar, $\frac{F^\odot}{4}$')
            flux_max_i = max([max(self.flux_plot['sw_up'][i]), max(self.flux_plot['lw_up'][i])])
            flux_min_i = -max([max(self.flux_plot['sw_down'][i]), max(self.flux_plot['lw_down'][i])])
            if self.ax_lims['flux'][1] > Animate.flux_plot_max_ax_lim > flux_max_i:
                current_flux_max = Animate.flux_plot_max_ax_lim
            else:
                current_flux_max = self.ax_lims['flux'][1]
            if self.ax_lims['flux'][0] < -5 and flux_min_i > -Animate.flux_plot_max_ax_lim:
                current_flux_min = -Animate.flux_plot_max_ax_lim
            else:
                current_flux_min = self.ax_lims['flux'][0]
            axs[-1].set_xlim((current_flux_min, current_flux_max))
            if self.log_axis:
                axs[-1].set_yscale('log')
            axs[-1].legend()
        t_years, t_days = t_years_days(self.t_plot[i])
        Title = "{:.0f}".format(t_years) + " Years and " + "{:.1f}".format(t_days) + " Days"
        ax.text(0.5, 1.01, Title, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)

    def animate_func_2d(self, i):
        self.plot2_info['cax'].cla()
        self.plot2_info['axColor'].clear()
        self.plot2_info['axTemp'].clear()
        im = self.plot2_info['axColor'].pcolormesh(self.plot2_info['X'],
                                                   self.plot2_info['Y'], self.T_plot[i], cmap='bwr')
        im.set_clim(self.ax_lims['T'])
        self.plot2_info['axColor'].axes.invert_yaxis()
        self.plot2_info['axColor'].set_ylim(self.ax_lims['p'])
        if self.log_axis:
            self.plot2_info['axColor'].set_yscale('log')
        self.plot2_info['axColor'].set_xticks(self.atmos.latitude)
        self.plot2_info['axTemp'].plot(self.atmos.latitude, self.T_plot[0][0], label='initial', linestyle='dotted')
        self.plot2_info['axTemp'].plot(self.atmos.latitude, self.T_plot[i][0], label='current')
        self.plot2_info['axTemp'].set_ylim(self.ax_lims['T'])
        self.plot2_info['axTemp'].set_xlabel('Latitude')
        self.plot2_info['axTemp'].set_ylabel('Surface Temperature / K')
        self.plot2_info['axTemp'].legend(loc='upper right')
        t_years, t_days = t_years_days(self.t_plot[i])
        Title = "{:.0f}".format(t_years) + " Years and " + "{:.1f}".format(t_days) + " Days"
        self.plot2_info['axColor'].text(0.5, 1.01, Title, horizontalalignment='center', verticalalignment='bottom',
                                        transform=self.plot2_info['axColor'].transAxes)
        cb = self.plot2_info['fig'].colorbar(im, cax=self.plot2_info['cax'])
        cb.set_label('Temperature / K')
        if self.compos_plot is not None:
            self.plot2_info['ax_compos'].clear()
            for key in self.compos_plot:
                self.plot2_info['ax_compos'].plot(self.compos_plot[key], self.atmos.p, label=key)
            if self.q_array is None:
                self.plot2_info['ax_compos'].set_xlabel(r'$\tau$')
            else:
                # log scale if plotting q
                self.plot2_info['ax_compos'].set_xlabel('Volume Mixing Ratio (ppmv)')
                self.plot2_info['ax_compos'].set_xscale('log')
            self.plot2_info['ax_compos'].set_xlim(self.ax_lims['compos'])
            if self.log_axis:
                self.plot2_info['ax_compos'].set_yscale('log')
            self.plot2_info['ax_compos'].axes.invert_yaxis()
            self.plot2_info['ax_compos'].legend(loc='upper right')
            self.plot2_info['ax_compos'].set_ylabel('Pressure / Pa')
            self.plot2_info['axTemp'].set_xticks(self.atmos.latitude)
            [l.set_visible(False) for (j, l) in enumerate(self.plot2_info['axColor'].xaxis.get_ticklabels())]
        else:
            self.plot2_info['axColor'].set_ylabel('Pressure / Pa')
            self.plot2_info['axColor'].set_xticks(self.atmos.latitude)
        [l.set_visible(False) for (j, l) in enumerate(self.plot2_info['axTemp'].xaxis.get_ticklabels())
         if j in self.plot2_info['miss_x_ticks']]
