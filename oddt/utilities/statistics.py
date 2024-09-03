import os
import MDAnalysis as mda
import pandas as pd



class statistics:
    def __init__(self):
        self.statistics()



    def statistics(self):
        os.chdir(folder)
        
        print('————————————————————\nSTATS CALCULATIONS\n————————————————————\n')
        top = 'solv.pdb'

        extension = top.split('.')[-1]
        if extension == 'psf':
            forcefield = 'CHARMM'
        else:
            forcefield = 'AMBER'

        u = mda.Universe(top)
        dimension = u.dimensions
        xyz = [['X', dimension[0]],['Y', dimension[1]],['Z', dimension[2]]]
        xyz_table = tabulate(xyz, tablefmt = 'pretty')

        if method == 'rt':
            components = ['resname WAT', 'protein', 'resname LIG', 'resname Na+', 'resname Cl-']
            
        elif method == 'ps':
            components = ['resname WAT', 'protein', 'resname Na+', 'resname Cl-']
        dictionary = []

        system = u.atoms
        n_resids = len(system.residues)
        n_atoms = system.n_atoms
        mass = system.total_mass(compound='group')
        entry = dict({'entry' : 'System', 'atoms' : n_atoms, 'molecules' : n_resids, 'mass [Da]' : mass, '[C] [mol/L]' : '-'})
        dictionary.append(entry)

        water_volume = 0
        water_C = 500 / 9   ### = 55.55555... M

        for mol in components:
            if mol == 'resname WAT':
                name = 'Water'
            elif mol == 'resname LIG':
                name = 'Ligand'
            elif mol == 'resname Na+':
                name = 'Na+'
            elif mol == 'resname Cl-':
                name = 'Cl-'
            elif mol == 'protein':
                name = 'Protein'

            sel = u.select_atoms(mol)

            if mol == 'protein':
                n_resids = 1
            else:
                n_resids = len(sel.residues)
            n_atoms = sel.n_atoms
            mass = sel.total_mass(compound='group')
            #moles = n_resids / (6.02214076 * (10 ** 23))
            if mol == 'resname WAT':
                C = water_C
                water_volume += n_resids / water_C
            else:
                C = n_resids / water_volume
            entry = dict({'entry' : name, 'atoms' : n_atoms, 'molecules' : n_resids, 'mass [Da]' : mass, '[C] [mol/L]' : C})
            dictionary.append(entry)


        df = pd.DataFrame.from_dict(dictionary, orient='columns')
        pd.set_option('display.float_format', '{:.4f}'.format)

        table = tabulate(df, headers = 'keys', tablefmt = 'psql', showindex=False)

        with open('stats.txt', 'w') as t:
            t.write(f'FORCE FIELD = {forcefield}\n\n')
            t.write('BOX DIMENSIONS [Å]\n')
            t.write(tabulate(xyz))
            t.write('\n\n\n')
            t.write(table)