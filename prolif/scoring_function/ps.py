import os
import importlib
import MDAnalysis as mda
import oddt
from oddt import fingerprints
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics



class score:
    def __init__(self, vars):
        self.__dict__ = vars


    def reference(self):
        if not os.path.exists('reference'):
            os.mkdir('reference')

        os.chdir('reference')

        self.contacts = self.find_contacts()
        ref_matrix = self.reference_matrix(self.contacts)

        update = {
                'contacts': self.contacts,
                'ref': ref_matrix
                }
        
        if self.are_rmsd_resids == True:
            update['df_resids'] = self.rmsd_resids

        else:
            update['df_resids'] = self.contacts['receptor']
            
        os.chdir('..')

        return update


    def find_contacts(self):
        contacts_module = importlib.import_module('..contacts', 'utilities.')
        resids = contacts_module.resids(self.__dict__)
        contacts = resids.residue_dict()
        return contacts


    def reference_matrix(self, contacts):
        if not os.path.exists('reference.pdb'):
            u = mda.Universe(self.solvpdb)
            dry = u.select_atoms('not resname WAT and not resname Na+ and not resname Cl-')

            with mda.Writer('reference.pdb', dry.n_atoms) as W:
                W.write(dry)

        mp_resids = []
        for r in contacts['receptor']:
            for l in contacts['ligand']:
                res = ([r, l])
                mp_resids.append(res)

        if not os.path.exists('reference_matrix.csv'):
            output = self.parallelizer.run(mp_resids, self.calc_ref, f'Calculating Reference Interaction Energy')

            ref = np.asarray(output)

            ref.tofile('reference_matrix.csv', sep=',')
                
        elif os.path.exists('reference_matrix.csv'):
            ref = np.loadtxt('reference_matrix.csv', delimiter=',')

        return ref


    def calc_ref(self, res_rec, res_lig):
        selection1 = f'resid {res_rec}'
        selection2 = f'resid {res_lig}'
        output_basename = f'interactionEnergy_{res_rec}_{res_lig}.dat'

        if not os.path.exists(f'{output_basename}'):
            vmdFile = f'{output_basename}.tcl'

            if not os.path.exists(f'{output_basename}'):
                with open(vmdFile, 'w') as f:
                    f.write(
f'''mol new {self.complprmtop}
mol addfile reference.pdb type pdb filebonds 1 autobonds 1 waitfor all
set prot [atomselect top "{selection1}"]
set ligand [atomselect top "{selection2}"]
global env
set Arch [vmdinfo arch]
set vmdEnv $env(VMDDIR)
puts $vmdEnv
source $vmdEnv/plugins/noarch/tcl/namdenergy1.4/namdenergy.tcl
namdenergy -exe {self.namd_path} -elec -vdw -sel $ligand $prot -ofile "{output_basename}" -tempname "{output_basename}_temp" -switch  7.5 -cutoff 9 -par {self.complprmtop}
quit'''
)
                os.system(f'{self.vmd_path} -dispdev text -e {vmdFile} > /dev/null 2>&1')

                os.remove(vmdFile)

        tot = pd.read_table(output_basename, sep='\s+')['Total'][0]
        out = round(tot, 4)

        os.system(f'rm {output_basename}')

        return out
            


    def score(self, topology, trajectory, temperature):
        if not os.path.exists('ie'):
            os.mkdir('ie')

        os.chdir('ie')

        mp_resids = []
        for r in self.contacts['receptor']:
            for l in self.contacts['ligand']:
                res = ([topology, trajectory, r, l])
                mp_resids.append(res)

        u = mda.Universe(topology, trajectory)
        ts = len(u.trajectory)
        output = self.parallelizer.run(mp_resids, self.calc_ie, f'Calculating Interaction Energy')

        os.chdir('..')

        outscore = []
        m = []

        for i in range(0,ts):
            l = []

            for o in output:
                l.append(o[i])
            
            array = np.asarray(l)

            l_plif_temp=[]

            l_plif_temp.append(self.ref)
            l_plif_temp.append(array)
            matrix = np.stack(l_plif_temp, axis=0)
            idx = np.argwhere(np.all(matrix[..., :] == 0, axis=0))
            matrix_dense = np.delete(matrix, idx, axis=1)
            x=matrix_dense[0].reshape(1,-1)
            y=matrix_dense[1].reshape(1,-1)
            sim_giovanni=float(sklearn.metrics.pairwise.cosine_similarity(x, y))
            sim = round(sim_giovanni * -1,2)

            outscore.append(sim)

            m.append(l)
            
        arr = np.array(m)

        arr.tofile(f'matrix_{temperature}.csv', sep = ',')

        return outscore


    
    def calc_ie(self, topology, trajectory, res_rec, res_lig):
        selection1 = f'resid {res_rec}'
        selection2 = f'resid {res_lig}'
        output_basename = f'interactionEnergy_{res_rec}_{res_lig}.dat'

        if not os.path.exists(f'{output_basename}'):
            vmdFile = f'{output_basename}.tcl'
            
            with open(vmdFile, 'w') as f:
                f.write(f'''mol new {topology}
        mol addfile {trajectory} type dcd  first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all
        set prot [atomselect top "{selection1}"]
        set ligand [atomselect top "{selection2}"]
        global env
        set Arch [vmdinfo arch]
        set vmdEnv $env(VMDDIR)
        puts $vmdEnv
        source $vmdEnv/plugins/noarch/tcl/namdenergy1.4/namdenergy.tcl
        namdenergy -exe {self.namd_path} -elec -vdw -sel $ligand $prot -ofile "{output_basename}" -tempname "{output_basename}_temp" -ts {self.dcdfreq} -timemult {self.timestep} -stride 1 -switch  7.5 -cutoff 9 -par {self.solvprmtop}
        quit''')

            os.system(f'vmd -dispdev text -e {vmdFile} > /dev/null 2>&1')

            os.remove(vmdFile)

        tot = pd.read_table(output_basename, sep='\s+')['Total']
        out = list(tot)

        os.system(f'rm -r {output_basename}')

        return out


