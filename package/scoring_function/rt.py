import os
import MDAnalysis as mda
import numpy as np
import sklearn.metrics
import prolif
from prolif import Molecule
from prolif.fingerprint import Fingerprint
from prolif.utils import to_countvectors
import time


class score:
    def __init__(self, vars):
        self.__dict__ = vars


    def reference(self):
        u = mda.Universe(self.solvprmtop, self.solvpdb)
        protein = u.select_atoms('nucleic')
        ligand = u.select_atoms('resname LIG')

        prot_ifp = Molecule.from_mda(protein)
        lig_ifp = Molecule.from_mda(ligand)

        fp = Fingerprint(count=True, vicinity_cutoff=100000000000000)
        ifp = fp.generate(lig_ifp, prot_ifp, residues='all', metadata=True)

        dataframe = prolif.to_dataframe({0: ifp}, fp.interactions, drop_empty=False)
        row = list(dataframe.itertuples(index=False))[0]
        array = np.array(row)


        # fp = Fingerprint(vicinity_cutoff=1000000000000000, count=True)
        # fp.run_from_iterable([lig_ifp], prot_ifp)

        # dataframe = fp.to_dataframe(drop_empty=False, count=True)
            
        # row = list(dataframe.itertuples(index=False))[0]
        # array = np.array(row)

        update = {'ref': array}
        return update

        

    def score(self, topology, trajectory, temperature):
        u = mda.Universe(topology, trajectory)
        
        mp_score = []

        for i,ts in enumerate(u.trajectory):
            mp_score.append([u, i])

        outscore = self.parallelizer.run(mp_score, self.calc_ifp, 'Calculating IFPs')

        return outscore



    def calc_ifp(self, u, i):
        u.trajectory[i]

        protein = u.select_atoms('nucleic')
        ligand = u.select_atoms('resname LIG')

        prot_ifp = Molecule.from_mda(protein)
        lig_ifp = Molecule.from_mda(ligand)

        fp = Fingerprint(count=True, vicinity_cutoff=100000000000000)
        ifp = fp.generate(lig_ifp, prot_ifp, residues='all', metadata=True)

        dataframe = prolif.to_dataframe({0: ifp}, fp.interactions, drop_empty=False)
        row = list(dataframe.itertuples(index=False))[0]
        array = np.array(row)


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


        return sim



