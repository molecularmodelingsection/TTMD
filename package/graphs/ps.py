import MDAnalysis as mda
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mplcolors
from scipy.interpolate import make_interp_spline
from scipy.stats import linregress
import pandas as pd
import seaborn as sns



class graphs:
    def __init__(self, vars):
        self.__dict__ = vars

        if self.dryer == 'yes':
            topology = self.complprmtop

        elif self.dryer == 'no':
            topology = self.solv.prmtop

        self.rmsd = self.calcRMSD(topology, self.final_dcd)

        self.titration_timeline()
        self.slope = self.titration_profile()

        self.matrix_profile()


    def titration_timeline(self):
        #### this function plots IFPcs and both backbone and ligand RMSD vs simulation time
        # plot IFPcs
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,9))
        x = self.time_list[:len(self.smooth_list)]
        y = np.array(self.smooth_list).astype(float)
        divnorm = mplcolors.TwoSlopeNorm(vmin=self.T_start, vcenter=(self.T_start + self.T_stop)/2, vmax=self.T_stop)
        axs[0].set_title('Interaction Energy Matrix Similarity')
        axs[0].set_xlabel('Time (ns)')
        axs[0].set_ylabel('IE Matrix$_{CS}$')
        axs[0].set_ylim(-1,0)
        # x_lim = ( temp_set[0][1] - temp_set[0][0] ) / temp_set[0][2] * temp_set[0][3] + temp_set[0][3]
        # axs[0].set_xlim(0,x_lim)
        axs[0].set_xlim(0,self.time_list[-1])

        s = axs[0].scatter(x, y, c=self.temperature_list[:len(self.smooth_list)], cmap='RdYlBu_r', norm=divnorm)

        cbar = add_colorbar_outside(s, self.temperature_list, ax=axs[0])
        cbar.set_label('Temperature (K)', rotation=270, labelpad=15)
        cbar.ax.set_yticklabels(np.array(self.temperature_list).astype('str'))


        # plot RMSD
        smooth = round(self.tot_frames*self.smooth/100)

        x = self.time_list[1:len(self.smooth_list)]

        y1 = self.rmsd[0]
        xnew = np.linspace(x[0], x[-1], smooth) 
        spl1 = make_interp_spline(x, y1, k=5)
        power_smooth1 = spl1(xnew)

        axs[1].plot(xnew, power_smooth1, color='seagreen', label='Chain A backbone')


        y2 = self.rmsd[1]
        spl2 = make_interp_spline(x, y2, k=5)
        power_smooth2 = spl2(xnew)

        axs[1].plot(xnew, power_smooth2, color='tomato', label='Chain B backbone')



        y3 = self.rmsd[2]
        spl3 = make_interp_spline(x, y3, k=5)
        power_smooth3 = spl3(xnew)

        axs[1].plot(xnew, power_smooth3, color='royalblue', label='BSite CM$_{distance}$')
        

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
        #### this function plots average IFPcs vs temperature
        fig, axs = plt.subplots(nrows=1, ncols=1)
        first_last_t = [self.T_start, self.T_stop]
        axs.set_xlim(first_last_t)
        axs.set_ylim(-1,0)
        axs.scatter(self.done_temp[:len(self.avg_list)], self.avg_list, c='royalblue')
        first_last_score = [-1.0, self.avg_list[-1]]

        f = np.poly1d(np.polyfit(first_last_t, first_last_score, 1))
        slope, intercept, r_value, p_value, std_err = linregress(first_last_t, first_last_score)
        axs.plot(self.temperature_list, f(self.temperature_list), color='tomato', ls='--', label="MS = {:.5f}".format(slope))
        axs.set_title('Titration Profile')
        axs.set_xlabel('Temperature (K)')
        axs.set_ylabel('Average IE Matrix$_{CS}$')
        axs.set_ylim(-1,0)
        axs.set_xlim(first_last_t)
        axs.legend()
        fig.savefig('titration_profile.png', dpi=300)

        return slope



    def calcRMSD(self, topology, trajectory):
        u = mda.Universe(topology, trajectory)

        selection_a = f'backbone and resid {self.receptor_chain[0]}:{self.receptor_chain[1]}'
        selection_b = f'backbone and resid {self.ligand_chain[0]}:{self.ligand_chain[1]}'
        chain_a = u.select_atoms(selection_a)
        chain_b = u.select_atoms(selection_b)

        R_a = rms.RMSD(u, u, select=selection_a, groupselections=[selection_b], weights_groupselections=['mass'], ref_frame=0).run()

        R_b = rms.RMSD(u, u, select=selection_b, ref_frame=0).run()

        rmsd_a = R_a.results.rmsd.T[2]
        rmsd_cmd = R_a.results.rmsd.T[3]
        rmsd_b = R_b.results.rmsd.T[2]

        return rmsd_a, rmsd_b, rmsd_cmd



    def matrix_profile(self):
        last_set = self.temperature[list(self.output.keys())[-1]]
        last_temp = last_set[0]
        last_len = last_set[1]

        last_step = f'MD/matrix_{last_temp}.csv'

        arr = np.genfromtxt(last_step, delimiter=',')
        last = np.split(arr, last_len/self.cfactor)[-1]

        diff = last - self.ref
        matrix = np.reshape(diff, (self.receptor_resnum, self.ligand_resnum))

        labels = []

        for c in self.contacts:
            chain = self.contacts[c]
            l_list = []
            for res in chain:
                resid = chain[res]
                label = resid['trueid'] + ' ' + str(resid['truenum'])
                l_list.append(label)

            labels.append(l_list)

        df = pd.DataFrame(matrix)

        cm, vmin, vmax = colorbar_quantile('RdBu_r', list(diff))  
            
        fig, ax = plt.subplots(facecolor='white')

        ax = sns.heatmap(df.astype(float), cmap=cm, center=0, vmin=vmin, vmax=vmax, yticklabels=1, xticklabels=1)
        cbar_axes = ax.figure.axes[-1]
        cbar_axes.set_ylabel('Interaction Energy gain (Kcal/mol)', rotation=270, labelpad=15)
        ax.tick_params(axis='x', rotation=90)
        ax.set_xticklabels(labels[0])
        ax.set_yticklabels(labels[1])
        ax.set_title('')
        ax.set_ylabel('Ligand Residue')
        ax.set_xlabel('Receptor Residue')
        plt.tight_layout()
        fig.savefig(f'matrix.png', dpi=300)
        




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



def colorbar_quantile(colorbar, color_list):
    cm = plt.cm.get_cmap(colorbar)
    sorted_list = np.sort(color_list)
    #### use numpy to calculate first quartile and third quartile
    vmin = np.nanquantile(sorted_list, 0.02)
    vmax = np.nanquantile(sorted_list, 0.98)

    return cm, vmin, vmax



def process_values(rr_filename_mean, index):

    res_dict = resid_dict()

    x = []
    y = []
    z = []
    color_list = []

    if index == 'all':
        for t in rr_filename_mean:
            rr = t[1]
            for arr in t[2]:
                rl = arr[0]
                v = arr[1]
                x.append(res_dict[rr])
                y.append(res_dict[rl])
                z.append(v)

        title = 'Pairwise Interaction Matrix'

    else:
        for t in rr_filename_mean:
            filename = t[0]
            rr = t[1]

            df = pd.read_csv(f'{_wd}/per_residue_matrix/{filename}', sep=',')
            
            for col in df.columns:
                for c in df[col]:
                    color_list.append(c)

                v = df[col][index]

                x.append(res_dict[rr])
                y.append(res_dict[col])
                z.append(v)

        title = f'Pairwise Interaction Matrix\n(Time = {time.time[index]})'

    ax_x = np.unique(x)
    ax_y = np.unique(y)

    ax = pd.DataFrame(columns=sorted(list(ax_x), key=lambda x: int(x.split(' ')[1])), index=sorted(list(ax_y), key=lambda x: int(x.split(' ')[1])))

    for a,b,c in zip(x,y,z):
        ax[a][b] = c

    return ax, color_list, title


def plot_matrix(rr_filename_mean, index):

    df, color_list, title = process_values(rr_filename_mean, index)

    cm, vmin, vmax = colorbar_quantile('RdBu_r', color_list)  
        
    fig, ax = plt.subplots(facecolor='white')

    ax = sns.heatmap(df.astype(float), cmap=cm, center=0, vmin=vmin, vmax=vmax, yticklabels=1, xticklabels=1)
    cbar_axes = ax.figure.axes[-1]
    cbar_axes.set_ylabel('Interaction Energy (Kcal/mol)', rotation=270, labelpad=15)
    ax.tick_params(labelsize=8, axis='x', rotation=45)
    ax.set_title(title)
    ax.set_ylabel('Ligand Residue')
    ax.set_xlabel('Receptor Residue')
    plt.tight_layout()
    fig.savefig(f'matrix_{index}.png', dpi=300)