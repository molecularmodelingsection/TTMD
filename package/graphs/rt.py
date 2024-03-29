import MDAnalysis as mda
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mplcolors
from scipy.interpolate import make_interp_spline
from scipy.stats import linregress
import importlib
import os



class graphs:
    def __init__(self, vars):
        self.__dict__ = vars

        if self.dryer == 'yes':
            topology = self.complprmtop

        elif self.dryer == 'no':
            topology = self.solvprmtop

        self.rmsd = calcRMSD(topology, self.final_dcd)

        if 'rmsd_resids' in self.__dict__.keys():
            self.bsite, self.avg_rmsd = bsite_rmsd(self.rmsd_resids, self.solvprmtop, self.done_temp, self.stop_range)
            self.rmsd_slope = self.rmsd_profile()
        else:
            self.rmsd_slope = 'Not calculated'
            self.bsite = 'None'

        self.titration_timeline()
        self.ms = self.titration_profile()

         



    def titration_timeline(self):
        #### this function plots IFPcs and both backbone and ligand RMSD vs simulation time
        # plot IFPcs
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,9))
        x = self.time_list[:len(self.smooth_list)]
        y = np.array(self.smooth_list).astype(float)
        divnorm = mplcolors.TwoSlopeNorm(vmin=self.T_start, vcenter=(self.T_start + self.T_stop)/2, vmax=self.T_stop)
        axs[0].set_title('Interaction Fingerprint Similarity')
        axs[0].set_xlabel('Time (ns)')
        axs[0].set_ylabel('IFP$_{CS}$')
        axs[0].set_ylim(-1,0)
        # x_lim = ( temp_set[0][1] - temp_set[0][0] ) / temp_set[0][2] * temp_set[0][3] + temp_set[0][3]
        # axs[0].set_xlim(0,x_lim)
        axs[0].set_xlim(0,self.time_list[-1])

        s = axs[0].scatter(x, y, c=self.temperature_list[:len(self.smooth_list)], cmap='RdYlBu_r', norm=divnorm)

        cbar = add_colorbar_outside(s, self.xticks, ax=axs[0])
        cbar.set_label('Temperature (K)', rotation=270, labelpad=15)
        # cbar.ax.set_yticklabels(np.array(self.temperature_list).astype('str'))


        # plot RMSD
        smooth = round(self.tot_frames*self.smooth/100)

        x = self.time_list[1:len(self.smooth_list)]

        y1 = self.rmsd[0]
        xnew = np.linspace(x[0], x[-1], smooth) 
        spl1 = make_interp_spline(x, y1, k=5)
        power_smooth1 = spl1(xnew)

        axs[1].plot(xnew, power_smooth1, color=self.colors[0], label='Backbone')


        y2 = self.rmsd[1]
        spl2 = make_interp_spline(x, y2, k=5)
        power_smooth2 = spl2(xnew)

        axs[1].plot(xnew, power_smooth2, color=self.colors[1], label='Ligand')
        
        if self.bsite != 'None':
            y3 = self.bsite
            spl3 = make_interp_spline(x, y3, k=5)
            power_smooth3 = spl3(xnew)

            axs[1].plot(xnew, power_smooth3, color=self.colors[2], label='BSite backbone')

        axs[1].set_ylabel('RMSD (Å)')
        axs[1].set_xlabel('Time (ns)')
        axs[1].set_title('RMSD')
        axs[1].set_xlim(0,self.time_list[-1])
        axs[1].set_ylim(0)
        axs[1].legend()
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.draw_without_rendering() #temporary fix for bug with colorbar label in matplotlb version 3.5.1
        fig.savefig('titration_timeline.png', dpi=300)




    def titration_profile(self):
        title = 'Titration Profile'
        ylim = [-1, 0]
        ylabel = 'Average IFP$_{CS}$'
        name = 'titration_profile'
        slope_start = -1
        
        module = importlib.import_module('..profile_graphs', __name__)
        slope = module.profile_graph(self.done_temp, self.avg_list, title, ylabel, name, self.colors, ylim=ylim, slope_start=slope_start)

        return slope



    def rmsd_profile(self):
        title = 'RMSD$_{Bsite Backbone}$ Profile'
        ylim = [0, None]
        ylabel = 'Average RMSD$_{B_site Backbone}$'
        name = 'rmsd_profile'
        slope_start = 0
        
        module = importlib.import_module('..profile_graphs', __name__)
        slope = module.profile_graph(self.done_temp, self.avg_rmsd, title, ylabel, name, self.colors, ylim=ylim, slope_start=slope_start)

        return slope



def calcRMSD(topology, trajectory):
    u = mda.Universe(topology, trajectory)
    R = rms.RMSD(u, u, select='backbone', groupselections=['resname LIG'], ref_frame=0).run()
    rmsd_backbone = R.results.rmsd.T[2]
    rmsd_ligand = R.results.rmsd.T[3]
    return rmsd_backbone, rmsd_ligand



def bsite_rmsd(rmsd_resids, topology, done_temp, stop_range):
    avg_rmsd = []
    plain_rmsd = []
    for temp in done_temp:
        trajectory = os.path.abspath(f'MD/swag_{temp}.dcd')
        u = mda.Universe(topology, trajectory)
        selection = ''
        for i,r in enumerate(rmsd_resids):
            if i == len(rmsd_resids) -1 :
                selection += f'resid {r}'
            else:
                selection += f'resid {r} or '

        n = int(len(u.trajectory)*stop_range/100)

        R = rms.RMSD(u, u, select=f'backbone and ({selection})', ref_frame=0).run()
        rmsd = list(R.results.rmsd.T[2])
        plain_rmsd.extend(rmsd)

        sum = 0
        for i in rmsd[-n:]:
            sum += i

        avg = sum / len(rmsd[-n:])
        avg_rmsd.append(avg)

    return plain_rmsd, avg_rmsd



def add_colorbar_outside(im, ticks, ax):
    fig = ax.get_figure()
    bbox = ax.get_position() #bbox contains the [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
    height = 0.4
    width = 0.01
    eps = 0.0 #margin between plot and colorbar
    pad = 0.0
    # [left most position, bottom position, width, height] of color bar.
    cax = fig.add_axes([bbox.x1 + eps, bbox.y0 + pad, width, height])#bbox.height])
    cbar = fig.colorbar(im, cax=cax, ticks=ticks)
    return cbar