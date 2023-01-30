import os
import MDAnalysis as mda
import numpy as np
import sklearn.metrics
import oddt
from oddt import fingerprints
from utilities.multiprocessing import parallelizer



class score:
    def __init__(self, vars):
        self.__dict__ = vars


    def reference(self):
        if not os.path.exists('reference_protein.pdb') or not os.path.exists('reference_ligand.pdb'):

            u = mda.Universe(self.solvpdb)
            protein = u.select_atoms('protein')
            ligand = u.select_atoms('resname LIG')

            with mda.Writer('reference_protein.pdb', protein.n_atoms) as W:
                W.write(protein)

            with mda.Writer('reference_ligand.pdb', ligand.n_atoms) as W:
                W.write(ligand)

        ref = self.ref_fingerprint('reference_protein.pdb', 'reference_ligand.pdb')

        update = {'ref': ref}
        return update



    def ref_fingerprint(self, protein_file, ligand_file):
        protein = next(oddt.toolkit.readfile('pdb', protein_file))
        protein.protein = True

        ligand = next(oddt.toolkit.readfile('pdb', ligand_file))
        
        fp = fingerprints.InteractionFingerprint(ligand, protein)

        return fp

        

    def score(self, topology, trajectory, temperature):
        if not os.path.exists('frame_pdbs'):
            os.mkdir('frame_pdbs')

        u = mda.Universe(topology, trajectory)

        mp_score = []

        for i,ts in enumerate(u.trajectory):
            mp_score.append([u, i])

        outscore = parallelizer.run(mp_score, self.calc_ifp, self.n_procs, 'Calculating IFPs')

        return outscore

        
    
    def calc_ifp(self, u, i):
        u.trajectory[i]

        u_protein = u.select_atoms('protein')
        protein_file = f'frame_pdbs/protein_{i}.pdb'

        u_ligand = u.select_atoms('resname LIG')
        ligand_file = f'frame_pdbs/ligand_{i}.pdb'

        with mda.Writer(protein_file, u_protein.n_atoms) as w:
            w.write(u_protein)

        with mda.Writer(ligand_file, u_ligand.n_atoms) as w:
            w.write(u_ligand)

        rdkitprotein = next(oddt.toolkit.readfile('pdb', protein_file))
        rdkitprotein.protein = True

        rdkitligand = next(oddt.toolkit.readfile('pdb', ligand_file))

        fp = fingerprints.InteractionFingerprint(rdkitligand, rdkitprotein)

        l_plif_temp=[]

        l_plif_temp.append(self.ref)
        l_plif_temp.append(fp)
        matrix = np.stack(l_plif_temp, axis=0)
        idx = np.argwhere(np.all(matrix[..., :] == 0, axis=0))
        matrix_dense = np.delete(matrix, idx, axis=1)
        x=matrix_dense[0].reshape(1,-1)
        y=matrix_dense[1].reshape(1,-1)
        sim_giovanni=float(sklearn.metrics.pairwise.cosine_similarity(x, y))
        sim = round(sim_giovanni * -1,2)

        os.system(f'rm -r {protein_file} {ligand_file}')

        return sim



