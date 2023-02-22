import importlib
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.stats import linregress
import numpy as np
import os


class graphs:
    def __init__(self, vars):
        self.__dict__ = vars

        if not os.path.exists('df'):
            os.mkdir('df')

        os.chdir('df')

        print('\n    Calculating Denaturating Factor')

        self.df_protein = self.intraprotein_df()
        self.df_prot_h2o = self.h2oprotein_df()

        os.chdir('..')



    def intraprotein_df(self):
        selection = ''
        for i,r in enumerate(self.rmsd_resids):
            if i == len(self.rmsd_resids) -1 :
                selection += f'resid {r}'
            else:
                selection += f'resid {r} or '

        ref_bonds = count_hbonds(self.solvprmtop, self.output['eq2']['dcd'], selection, 'protein', 100, basename='ref_prot_hbonds')
        
        temp_hbonds = []
        for i in self.done_temp:
            hbonds = count_hbonds(self.solvprmtop, f'../MD/swag_{i}.dcd', selection, 'protein', self.stop_range, basename=f'prot_hbonds', i=i)
            df = 1 - (hbonds/ref_bonds)
            temp_hbonds.append(df)

        title = 'Intraproteic DF Profile'
        ylabel = 'Average Hbonds loss'
        name = '../df_profile'
        slope_start = 0
        ylim = [0, None]

        module = importlib.import_module('..profile_graphs', __name__)
        slope = module.profile_graph(self.done_temp, temp_hbonds, title, ylabel, name, self.colors, ylim=ylim, slope_start=slope_start)

        return slope



    def h2oprotein_df(self):
        residues = ''

        for i,r in enumerate(self.rmsd_resids):
            if i == len(self.rmsd_resids) -1 :
                residues += f'resid {r}'
            else:
                residues += f'resid {r} or '

        sel1 = residues
        sel2 = f'(resname WAT and same residue as within 5 of ({residues}))'

        ref_bonds = count_hbonds(self.solvprmtop, self.output['eq2']['dcd'], sel1, sel2, 100, basename=f'ref_wat_hbonds')        

        temp_hbonds = []

        for i in self.done_temp:
            hbonds = count_hbonds(self.solvprmtop, f'../MD/swag_{i}.dcd', sel1, sel2, self.stop_range, basename=f'wat_hbonds', i=i)
            temp_hbonds.append(hbonds)


        title = 'Protein-Water DF Profile'
        ylabel = 'Average Hbonds gain'
        name = '../df_h2o_profile'
        slope_start = 0
        ylim = [0, None]

        module = importlib.import_module('..profile_graphs', __name__)
        slope = module.profile_graph(self.done_temp, temp_hbonds, title, ylabel, name, self.colors, slope_start=slope_start, ylim=ylim)

        return slope



def count_hbonds(topology, trajectory, sel1, sel2, stop_range, i=None, basename=None):
    filename = basename
    if i != None:
        filename += f'_{i}'
    
    u = mda.Universe(topology, trajectory)
    n = int(len(u.trajectory)*stop_range/100)

    if not os.path.exists(filename):
        hbonds = f'''mol delete all;
    mol load parm7 {topology} dcd {trajectory}
    set protein [atomselect top "{sel1}"]
    set lig [atomselect top "{sel2}"]
    package require hbonds
    hbonds -sel1 $protein -sel2 $lig -writefile yes -dist 3.0 -ang 30 -outfile {filename} -type all
    quit'''

        with open('hbonds.tcl','w') as f:
            f.write(hbonds)

        os.system('vmd -dispdev text -e hbonds.tcl > /dev/null 2>&1')


    arr = np.loadtxt(filename, delimiter=' ')
    l = list(arr.T[1])[-n:]
    s = 0
    for i in l:
        s += i

    avg = s / len(l)

    return avg