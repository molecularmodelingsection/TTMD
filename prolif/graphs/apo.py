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
        self.ms = 'Not available with this method'
        # self.ms = self.titration_profile()

         

    def titration_timeline(self):
        #### this function plots IFPcs and both backbone and ligand RMSD vs simulation time
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,9))

        # plot RMSD
        smooth = round(self.tot_frames*self.smooth/100)

        x = self.time_list[1:len(self.smooth_list)]
        y1 = self.rmsd
        xnew = np.linspace(x[0], x[-1], smooth) 
        spl1 = make_interp_spline(x, y1, k=5)
        power_smooth1 = spl1(xnew)

        axs.plot(xnew, power_smooth1, color=self.colors[0], label='Backbone')
        
        if self.bsite != 'None':
            y2 = self.bsite
            spl2 = make_interp_spline(x, y2, k=5)
            power_smooth2 = spl2(xnew)

            axs.plot(xnew, power_smooth2, color=self.colors[1], label='BSite backbone')

        axs.set_ylabel('RMSD (Å)')
        axs.set_xlabel('Time (ns)')
        axs.set_title('RMSD')
        axs.set_xlim(0,self.time_list[-1])
        axs.set_ylim(0)
        axs.legend()
        axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.tight_layout()
        fig.draw_without_rendering() #temporary fix for bug with colorbar label in matplotlb version 3.5.1
        fig.savefig('titration_timeline.png', dpi=300)




    # def titration_profile(self):
    #     title = 'Titration Profile'
    #     ylim = [-1, 0]
    #     ylabel = 'Average IFP$_{CS}$'
    #     name = 'titration_profile'
    #     slope_start = -1
        
    #     module = importlib.import_module('..profile_graphs', __name__)
    #     slope = module.profile_graph(self.done_temp, self.avg_list, title, ylabel, name, self.colors, ylim=ylim, slope_start=slope_start)

    #     return slope



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
    R = rms.RMSD(u, u, select='backbone', ref_frame=0).run()
    rmsd_backbone = R.results.rmsd.T[2]
    return rmsd_backbone



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



# def add_colorbar_outside(im, ticks, ax):
#     fig = ax.get_figure()
#     bbox = ax.get_position() #bbox contains the [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
#     height = 0.4
#     width = 0.01
#     eps = 0.0 #margin between plot and colorbar
#     pad = 0.0
#     # [left most position, bottom position, width, height] of color bar.
#     cax = fig.add_axes([bbox.x1 + eps, bbox.y0 + pad, width, height])#bbox.height])
#     cbar = fig.colorbar(im, cax=cax, ticks=ticks)
#     return cbar