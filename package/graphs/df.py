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

        ref_bonds = count_hbonds(self.solvprmtop, self.output['eq2']['dcd'], selection, 'nucleic', 100, 'ref_prot_hbonds') #adodaro
        
        mp_hbonds = []
        
        for i in self.done_temp:
            mp_hbonds.append([self.solvprmtop, f'../MD/swag_{i}.dcd', selection, 'nucleic', self.stop_range, f'prot_hbonds_{i}']) #adodaro
            
        temp_hbonds = self.parallelizer.run(mp_hbonds, count_hbonds, '      Calculating Receptor DF') #adodaro
        
        hbonds = []
        for v in temp_hbonds:
            df = 1 - (v/ref_bonds)
            hbonds.append(df)

        title = 'Intraproteic DF Profile'
        ylabel = 'Average Hbonds loss'
        name = '../df_profile'
        slope_start = 0
        ylim = [None, None]

        module = importlib.import_module('..profile_graphs', __name__)
        slope = module.profile_graph(self.done_temp, hbonds, title, ylabel, name, self.colors, ylim=ylim, slope_start=slope_start)

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

        ref_bonds = count_hbonds(self.solvprmtop, self.output['eq2']['dcd'], sel1, sel2, 100, 'ref_wat_hbonds')        

        mp_hbonds = []
        
        for i in self.done_temp:
            mp_hbonds.append([self.solvprmtop, f'../MD/swag_{i}.dcd', sel1, sel2, self.stop_range, f'wat_hbonds_{i}'])

        temp_hbonds = self.parallelizer.run(mp_hbonds, count_hbonds, '      Calculating H2O-Receptor DF') #adodaro
        
        title = 'Nucleic-Water DF Profile' #adodaro
        ylabel = 'Average Hbonds gain'
        name = '../df_h2o_profile'
        slope_start = 0
        ylim = [None, None]

        module = importlib.import_module('..profile_graphs', __name__)
        slope = module.profile_graph(self.done_temp, temp_hbonds, title, ylabel, name, self.colors, slope_start=slope_start, ylim=ylim)

        return slope



def count_hbonds(topology, trajectory, sel1, sel2, stop_range, filename):
    u = mda.Universe(topology, trajectory)
    n = int(len(u.trajectory)*stop_range/100)
      
    if not os.path.exists(filename):        #adodaro set protein & sel1 $protein
        hbonds = f'''mol delete all;
mol load parm7 {topology} dcd {trajectory}
set nucleic [atomselect top "{sel1}"]
set lig [atomselect top "{sel2}"]
package require hbonds
hbonds -sel1 $nucleic -sel2 $lig -writefile yes -dist 3.0 -ang 30 -outfile {filename} -type all
quit'''

        with open(f'{filename}.tcl','w') as f:
            f.write(hbonds)

        os.system(f'vmd -dispdev text -e {filename}.tcl > /dev/null 2>&1')
        os.system(f'rm {filename}.tcl')


    arr = np.loadtxt(filename, delimiter=' ')
    l = list(arr.T[1])[-n:]
    s = 0
    for i in l:
        s += i

    avg = s / len(l)

    return avg
