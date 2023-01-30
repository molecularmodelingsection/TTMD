import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from MDAnalysis.analysis import align
from utilities.multiprocessing import parallelizer



class wrapping:
    def __init__(self, vars):
        self.__dict__ = vars



    def run(self, topology, trajectory, wrap_trj):
        blocks = trajectory_blocks(topology, trajectory, self.n_procs)[0]
        wrapped_blocks = parallelizer.run(blocks, wrap_blocks, self.n_procs, '    Wrapping blocks')
        self.merge_trj(topology, wrapped_blocks, wrap_trj, remove=True)



    def merge_trj(self, topology, trj_list, trj_name, remove=bool):
        u = mda.Universe(topology, trj_list)

        with mda.Writer(trj_name, u.atoms.n_atoms) as W:
            for ts in u.trajectory:
                W.write(u.atoms)
        
        if remove == True:
            for trj in trj_list:
                os.system(f'rm {trj}')



    def wrap_equil2(self, topology, trajectory, wrap_trj):
        wrap_protocol = f'''mol new {topology}
mol addfile {trajectory} type dcd first 1 last -1 step 1 filebonds 1 autobonds 1 waitfor all
package require pbctools
pbc wrap -all -compound res -center bb -centersel "none"
animate write dcd {wrap_trj}
quit'''

        with open('wrap.tcl', 'w') as f:
            f.write(wrap_protocol)

        os.system(f'vmd -dispdev text -e wrap.tcl')



    def dry_trj(self, trajectory, set):
        temp, length = set
        name = f'dry_{temp}.dcd'

        if not os.path.exists(name):
            print(f'\n    {name} doesn\'t exists')
            check = False

        else:
            check = self.check_trj_len.check(self.complprmtop, name, length)
            

        if check == False:
            u = mda.Universe(self.solvprmtop, trajectory)
            udry = u.select_atoms('not resname WAT and not resname Cl- and not resname Na+')

            with mda.Writer(name, udry.n_atoms) as W:
                for ts in u.trajectory:
                    W.write(udry.atoms)

        return name
        


def wrap_blocks(first, last, topology, trajectory):

    u = mda.Universe(topology, trajectory)
    u.trajectory[first:last]

    ref = mda.Universe(topology)
    ref_dry = ref.select_atoms('not resname WAT and not resname Na+ and not resname Cl-')

    try:
        u.fragments()
    except Exception:
        u.atoms.guess_bonds()

    system_dry = u.select_atoms('not resname WAT and not resname Na+ and not resname Cl-')
    solvent = u.select_atoms('resname WAT or resname Cl- or resname Na+')

    transforms = [trans.unwrap(u.atoms),
                trans.center_in_box(system_dry, center='geometry', wrap=False),
                trans.wrap(solvent, compound='residues'),
                ]
    u.trajectory.add_transformations(*transforms)

    block_name = f'block_{first}_{last}.dcd'
    with mda.Writer(block_name, u.atoms.n_atoms) as W:
        for ts in u.trajectory[first:last]:
            old_rmsd, new_rmsd = align.alignto(u, ref, select='protein and backbone', weights='mass')
            W.write(u.atoms)

    return block_name



def trajectory_blocks(topology, trajectory, n_procs):
    u = mda.Universe(topology, trajectory)
    n_frames = u.trajectory.n_frames
    frames_per_block = n_frames // n_procs
    blocks = [[i * frames_per_block, (i + 1) * frames_per_block, topology, trajectory]for i in range(n_procs - 1)]
    blocks.append([(n_procs - 1) * frames_per_block, n_frames, topology, trajectory])
    return blocks, n_frames



