###############################################################################
### SET PARAMETERS

### TEMPERATURE SET LIST:
# [[t_a1, t_z1, interval1, step1], [t_a2, t_z2, interval2, step2], [...]] (int format)
temp_set = [[300, 450, 10, 10]] #[t_start, t_end, t_step, step_len]

### COMPUTER SETTINGS
device = 0      ### GPU device ID (int format)
n_procs = 4     ### cores number for analysis modules (int format)

protein_name = 'protein.pdb'    ### protein filename (.pdb)
ligand_name = 'ligand.mol2'     ### ligand filename (.mol2)
ligand_charge = 0               ### ligand charge (int format)

### EXTERNAL DEPENDENCIES PATH
wordom = '/odex/bin/wordom'
vmd = '/odex/bin/vmd'

### water padding around protein (Å)
padding = 15

dryer = 'yes'       ### if 'yes': dry final merged trj (output without water and ions)
                        # N.B. a dryed trj needs less disk space,
                            # while water coordinates can be retrieved from
                            # conserved run_*.dcd and swag_*.dcd trajectories
                    ### if 'no': final trj with water and ions

### components selection for statistics
components = ['resname WAT', 'protein', 'resname LIG', 'resname Na+', 'resname Cl-']

### RESTART FEATURE (EXPERIMENTAL!!!)
resume = True       ### if True: resume simulation
                    ### if False: reset folders and restart from system preparation

### comment MAIN row(s) to skip
def MAIN():
    ### PREPARATORY STEPS   
    prepare_system()
    statistics()
    gpu_info()
    equil()
    ### TITRATION BLOCK
    thermic_titration()
    final_merge_trj()
    titration_timeline()
    titration_profile()

### LAUNCHING TTMD
# check temperature set list and computer settings.
# starting folder set-up: protein file (.pdb), ligand file (.mol2)
# launch: 'python3 TTMD_file.py'
# N.B. this script works with python3

###############################################################################

min_steps  = 500     ### minimization steps with the conjugate gradient algorithm before equil1
equil1_len = 0.1     ### equil1 duration (ns)
equil2_len = 0.5     ### equil2 duration (ns)

timestep = 2         # integration timestep in fs
dcdfreq = 10000      # frequency at which a trajectory frame is saved
stride = 1           # stride to apply to final trajectory

smooth = 200        ### smoothing factor in final graphs

###############################################################################

header = '''
###########################################################################
                                                                           
                ███         ███       ▄▄▄▄███▄▄▄▄   ████████▄                
            ▀█████████▄ ▀█████████▄ ▄██▀▀▀███▀▀▀██▄ ███   ▀███            
               ▀███▀▀██    ▀███▀▀██ ███   ███   ███ ███    ███               
                ███   ▀     ███   ▀ ███   ███   ███ ███    ███              
                ███         ███     ███   ███   ███ ███    ███             
                ███         ███     ███   ███   ███ ███    ███             
                ███         ███     ███   ███   ███ ███   ▄███             
               ▄████▀      ▄████▀    ▀█   ███   █▀  ████████▀                 
                             
                                                                @smenin

    "Qualitative Estimation of Protein-Ligand Complex Stability through    
         Thermal Titration Molecular Dynamics (TTMD) Simulations."  
         
            Pavan M., Menin S., Bassani D., Sturlese M., Moro S.
            @Molecular Modeling Section, University of Padova
            
###########################################################################                        
'''

###############################################################################


import os
import sys
import glob
from tabulate import tabulate
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mplcolors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import oddt
from oddt import fingerprints
import MDAnalysis as mda
import MDAnalysis.transformations as trans
from MDAnalysis.analysis import align
from MDAnalysis.analysis import rms
import sklearn.metrics
from sklearn.metrics import pairwise_distances
from scipy.interpolate import make_interp_spline, BSpline
from scipy.stats import linregress
import multiprocessing
import time
import tqdm


np.set_printoptions(threshold=sys.maxsize)
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

folder = os.getcwd()

done_temperature = []

ref_fp = object

T_start = temp_set[0][0]
T_stop = temp_set[-1][1]

conversion_factor = timestep * dcdfreq * stride / 1000000

script_name = os.path.basename(__file__)

###############################################################################
### RESUME FUNCTIONS

resume_check = False

def gpu_check():
    os.system(f'nvidia-smi -i {device} --query-gpu=name --format=csv,noheader -f gpu_check.info')
    with open('gpu_check.info', 'r') as f:
        gpu_name = f.readlines()[0]
    os.system('rm gpu_check.info')
    return gpu_name



def gpu_info():
    global resume_check
    os.chdir(folder)
    
    if resume == True:
    
        if os.path.exists(f'{folder}/gpu.info'):

            with open(f'{folder}/gpu.info', 'r') as i:
                gpu = i.readlines()[0]
            
            current_gpu = gpu_check()
            c_gpu = current_gpu.rstrip('\n')

            if current_gpu == gpu:
                resume_check = True
                print(f'{c_gpu} = {gpu}\n——————————\nResume ON\n——————————\n')

            else:
                with open(f'{folder}/gpu.info', 'w') as i:
                    i.write(current_gpu)
                print(f'{c_gpu} != {gpu}\n——————————\nResume OFF\n——————————\n')

        else:
            resume_check = True
            
            with open(f'{folder}/gpu.info', 'w') as i:
                current_gpu = gpu_check()
                i.write(current_gpu)

    elif resume == False:
        files = glob.glob('*')
        for f in files:
            if f == ligand_name or f == protein_name or f == script_name:
                continue
            else:
                os.system(f'rm -r {f}')
                print(f'{f} removed')

        with open(f'{folder}/gpu.info', 'w') as i:
            i.write(gpu_check().rstrip('/n'))

###############################################################################
### MULTIPROCESSING FUNCTION AND MODULES

class parallelizer(object):
    ### base class for multiprocessing
    def __init__(self, args, func, num_procs, desc):
        ### function initialization
        self.num_procs = n_procs
        self.args = args
        self.func = func
        self.desc = desc

    def start(self):
        pass

    def end(self):
        pass

    def run(args, func, num_procs, desc):
        return MultiThreading(args, func, num_procs, desc)
        ### run takes 4 arguments:
            # list of tup(args) for each spawned process
            # name of the function to be multiprocessed
            # number of process to spwan
            # description for the progression bar



def MultiThreading(args, func, num_procs, desc):
    results = []
    tasks = []
    for index,item in enumerate(args):
        task = (index, (func, item))
        ### every queue objects become indexable
        tasks.append(task)
    ### step needed to rethrieve correct results order
    results = start_processes(tasks, num_procs, desc)
    return results



def start_processes(inputs, num_procs, desc):
    ### this function effectively start multiprocess
    task_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()
    ### queue objects to manage to do args and results

    for item in inputs:
        ### inputs = [(index, (function_name, arg)), ...]
        task_queue.put(item)
        ### every item is passed to the task_queue

    pbar = tqdm.tqdm(total=len(inputs), desc=desc)
    ### progress bar is initialized
    
    for i in range(n_procs):
        p = multiprocessing.Process(target=worker, args=(task_queue, done_queue)).start()
        ### spawn (n_proc) worker function, that takes queue objects as args

    results = []
    for i in range(len(inputs)):
        results.append(done_queue.get())
        pbar.update(1)
        ### done_queue and progress bar update for each done object

    for i in range(num_procs):
        task_queue.put("STOP")
        ### to exit from each spawned process when task queue is empty

    results.sort(key=lambda tup: tup[0])    ### sorting of args[a] - args[z] results
    return [item[1] for item in map(list, results)] ### return a list with sorted results



def worker(input, output):
    ### input, output = task_queue, done_queue lists
    for seq, job in iter(input.get, "STOP"):
        ### seq = object index
        ### job = function name, args for function

        func, args = job
        result = func(*args)
        ### function is executed and return a result value

        ret_val = (seq, result)
        ### assign index to result value

        output.put(ret_val)
        ### (index, result) object is put in done_queue

###############################################################################
### MULTIPROCESSING FUNCTIONS

def wrap_blocks(*args):
    first = list(args[0])[0]
    last = list(args[1])[0]
    topology = list(args[2])[0]
    trajectory = list(args[3])[0]

    u = mda.Universe(topology, trajectory)
    u.trajectory[first:last]

    ref = mda.Universe('solv.pdb') 
    protein = u.select_atoms('protein')
    not_protein = u.select_atoms('not protein')
    whole = u.select_atoms('all')
    transforms = [trans.unwrap(whole),
                    trans.center_in_box(protein, wrap=False),
                    trans.wrap(whole, compound='residues')]
    u.trajectory.add_transformations(*transforms)
    block_name = f'block_{first}_{last}.dcd'
    with mda.Writer(block_name, u.atoms.n_atoms) as W:
        for ts in u.trajectory[first:last]:
            old_rmsd, new_rmsd = align.alignto(u, ref, select='protein and backbone', weights='mass')
            W.write(u.atoms)
    return block_name



def trajectory_blocks(topology, trajectory):
    u = mda.Universe(topology, trajectory)
    n_frames = u.trajectory.n_frames
    frames_per_block = n_frames // n_procs
    blocks = [[(i * frames_per_block,), ((i + 1) * frames_per_block,), (topology,), (trajectory,)]for i in range(n_procs - 1)]
    blocks.append([((n_procs - 1) * frames_per_block,), (n_frames,), (topology,), (trajectory,)])
    return blocks, n_frames



def merge_trj(trj_list, trj_name, remove=bool):
    with open('merge_trj.txt', 'w') as f:
        for trj in trj_list:
            f.write(trj + '\n')
    
    os.system(f'{wordom} -itrj merge_trj.txt -otrj {trj_name}')
    
    if remove == True:
        for trj in trj_list:
            os.system(f'rm {trj}')

    os.system('rm merge_trj.txt')



def write_frame(*args):
    index_list = list(args[0])
    index = index_list[0]
    topology_list = list(args[1])
    topology = topology_list[0]
    trajectory_list = list(args[2])
    trajectory = trajectory_list[0]

    u = mda.Universe(topology, trajectory)
    u.trajectory[index]
    protein = u.select_atoms('protein')
    ligand = u.select_atoms('resname LIG')

    protein_name = 'frame_PDBs/frame_protein_' + str(index) + '.pdb'
    ligand_name = 'frame_PDBs/frame_ligand_' + str(index) + '.pdb'

    with mda.Writer(protein_name, protein.n_atoms) as W:
        W.write(protein)
    with mda.Writer(ligand_name, ligand.n_atoms) as W:
        W.write(ligand)
    
    return protein_name, ligand_name



def ifp(*args):
    protein_file = list(args)[0]
    protein = next(oddt.toolkit.readfile('pdb', protein_file))
    protein.protein = True

    ligand_file = list(args)[1]
    ligand = next(oddt.toolkit.readfile('pdb', ligand_file))
    
    fp = fingerprints.InteractionFingerprint(ligand, protein)

    l_plif_temp=[]
    l_plif_temp.append(ref_fp)
    l_plif_temp.append(fp)
    matrix = np.stack(l_plif_temp, axis=0)
    idx = np.argwhere(np.all(matrix[..., :] == 0, axis=0))
    matrix_dense = np.delete(matrix, idx, axis=1)
    x=matrix_dense[0].reshape(1,-1)
    y=matrix_dense[1].reshape(1,-1)
    sim_giovanni=float(sklearn.metrics.pairwise.cosine_similarity(x, y))
    sim = round(sim_giovanni * -1,2)
    return sim



def calculate_ifp(topology, trajectory, temperature):
    u = mda.Universe(topology, trajectory)
    ts_list = []
    for ts in u.trajectory:
        ts_list.append([(ts.frame,), (topology,), (trajectory,)])
    
    if not os.path.exists("frame_PDBs"):
        os.mkdir("frame_PDBs")

    pdb_list = parallelizer.run(ts_list, write_frame, n_procs, 'Writing PDBs')
    ifp_list = parallelizer.run(pdb_list, ifp, n_procs, 'Calculating fingerprints')
    
    with open(f'sim_{temperature}.dat', 'w') as sim:
        for x in ifp_list:
            sim.write(f'{x}\n')
    
    return ifp_list
        


def dry_trj(*args):
    first = list(args[0])[0]
    last = list(args[1])[0]
    topology = list(args[2])[0]
    trajectory = list(args[3])[0]

    u = mda.Universe(topology, trajectory)
    dry_sel = u.select_atoms("not resname WAT and not resname Na+ and not resname Cl-")
    
    block_name = f'block_{first}_{last}.dcd'
    with mda.Writer(block_name, dry_sel.n_atoms) as W:
        for ts in u.trajectory[first:last]:
            W.write(dry_sel)
    return block_name



###############################################################################

def prepare_system():
    os.chdir(folder)
    print('————————————————————\nSystem preparation\n————————————————————\n')

    with open("complex.in", 'w') as f:
        f.write(f"""source leaprc.protein.ff14SB
source leaprc.water.tip3p
source leaprc.gaff
loadamberprep ligand.prepi
loadamberparams ligand.frcmod
loadoff atomic_ions.lib
loadamberparams frcmod.ionsjc_tip3p
PROT = loadpdb {protein_name}
LIG = loadmol2 ligand_charged.mol2
COMPL = combine{{PROT LIG}}
saveAmberParm LIG ligand.prmtop ligand.inpcrd
saveAmberParm PROT protein.prmtop protein.inpcrd
saveAmberParm COMPL complex.prmtop complex.inpcrd
solvatebox COMPL TIP3PBOX {padding}
savepdb COMPL solv.pdb
saveamberparm COMPL solv.prmtop solv.inpcrd
quit
    """)

    with open("determine_ions_fixed.vmd", 'w') as f:
        f.write("""set saltConcentration 0.154
mol delete all
mol load parm7 solv.prmtop pdb solv.pdb 
set sel [atomselect top "water and noh"];
set nWater [$sel num];
$sel delete
if {$nWater == 0} {
    error "ERROR: Cannot add ions to unsolvated system."
    exit
}
set all [ atomselect top all ]
set charge [measure sumweights $all weight charge]
set intcharge [expr round($charge)]
set chargediff [expr $charge - $intcharge]
if { ($chargediff < -0.01) || ($chargediff > 0.01) } {
    error "ERROR: There is a problem with the system. The system does not seem to have integer charge."
    exit
}
puts "System has integer charge: $intcharge"
set cationStoich 1
set anionStoich 1
set cationCharge 1
set anionCharge -1
set num [expr {int(0.5 + 0.0187 * $saltConcentration * $nWater)}]
set nCation [expr {$cationStoich * $num}]
set nAnion [expr {$anionStoich * $num}]
if { $intcharge >= 0 } {
    set tmp [expr abs($intcharge)]
    set nCation [expr $nCation - round($tmp/2.0)]
    set nAnion  [expr $nAnion + round($tmp/2.0)] 
    if {$intcharge%2!=0} {
    set nCation [expr $nCation + 1]}
    puts "System charge is positive, so add $nCation cations and $nAnion anions"
} elseif { $intcharge < 0 } {
    set tmp [expr abs($intcharge)]
    set nCation [expr $nCation + round($tmp/2.0)]
    set nAnion  [expr $nAnion - round($tmp/2.0)]
    if {$intcharge%2!=0} { 
    set nAnion [expr $nAnion + 1]}
    puts "System charge is negative, so add $nCation cations and $nAnion anions"
}
if { [expr $intcharge + $nCation - $nAnion] != 0 } {
    error "ERROR: The calculation has gone wrong. Adding $nCation cations and $nAnion will not result in a neutral system!"
    exit
}
puts "\n";
puts "Your system already has the following charge: $intcharge"
puts "Your system needs the following ions to be added in order to be \
neutralized and have a salt concentration of $saltConcentration M:"
puts "\tCations of charge $cationCharge: $nCation"
puts "\tAnions of charge $anionCharge: $nAnion"
puts "The total charge of the system will be [expr $intcharge + $nCation - $nAnion]."
puts "\n";
exit""")



    os.system(f"antechamber -fi mol2 -i {ligand_name} -o ligand_charged.mol2 -fo mol2 -nc {ligand_charge} -c bcc -pf y -rn LIG")
    os.system("antechamber -fi mol2 -i ligand_charged.mol2 -o ligand.prepi -fo prepi -pf y")
    os.system("parmchk2 -i ligand.prepi -f prepi  -o ligand.frcmod")
    os.system("tleap -f complex.in")
    os.system(f"{vmd} -dispdev text -e determine_ions_fixed.vmd > ion.log")

    with open("ion.log",'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Cations of charge 1' in line:
                cations = str(line.split(':')[1].strip())
            elif 'Anions of charge -1' in line:
                anions = str(line.split(':')[1].strip())
            else:
                pass
    os.system('rm determine_ions_fixed.vmd ion.log')



    with open("complex.in", 'w') as f:
        f.write(f"""source leaprc.protein.ff14SB
source leaprc.water.tip3p 
source leaprc.gaff
loadamberprep ligand.prepi
loadamberparams ligand.frcmod
loadoff atomic_ions.lib
loadamberparams frcmod.ionsjc_tip3p
PROT = loadpdb {protein_name}
LIG = loadmol2 ligand_charged.mol2
COMPL = combine{{PROT LIG}}
saveAmberParm LIG ligand.prmtop ligand.inpcrd
saveAmberParm PROT protein.prmtop protein.inpcrd
saveAmberParm COMPL complex.prmtop complex.inpcrd
solvatebox COMPL TIP3PBOX {padding}
addIonsRand COMPL Na+ {cations} Cl- {anions} 5
savepdb COMPL solv.pdb
saveamberparm COMPL solv.prmtop solv.inpcrd
quit
""")

    os.system(f"tleap -f complex.in")



    with open("check_charge.vmd", 'w') as f:
        f.write("""mol load parm7 solv.prmtop pdb solv.pdb
set all [atomselect top all]
set curr_charge [measure sumweights $all weight charge]
puts [format "\nCurrent system charge is: %.3f\n" $curr_charge]
exit""")

    os.system(f"{vmd} -dispdev text -e check_charge.vmd > charge.log")

    with open("charge.log",'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Current system charge is: '):
                end = str(line.split(':')[1].strip())
        if end.startswith("0.") or end.startswith("-0."):
            pass
        else:
            exit("Error: system charge is not 0!")
    os.system("rm check_charge.vmd charge.log")

    if not os.path.exists('equil1'):
        os.mkdir("equil1")
    if not os.path.exists('equil2'):
        os.mkdir("equil2")
    if not os.path.exists('MD'):
        os.mkdir("MD")

    os.system("cp solv.pdb equil1/")
    os.system("cp solv.prmtop equil1/")
    os.system("cp solv.pdb equil2/")
    os.system("cp solv.prmtop equil2/")
    os.system("cp solv.pdb MD/")
    os.system("cp solv.prmtop MD/")



def statistics():
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

    components = ['resname WAT', 'protein', 'resname LIG', 'resname Na+', 'resname Cl-']
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



def equil1(restart):
    
    with open("get_celldimension.vmd", 'w') as f:
        f.write("""mol delete all;
mol load parm7 solv.prmtop pdb solv.pdb
set all [atomselect top all];
set box [measure minmax $all];
set min [lindex $box 0];
set max [lindex $box 1];
set cell [vecsub $max $min];
put "celldimension $cell"
quit""")

    os.system(f"{vmd} -dispdev text -e get_celldimension.vmd > celldimension.log")

    with open("celldimension.log",'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('celldimension '):
                dimension = str(line.split(' ')[1]) + ' ' + str(line.split(' ')[2]) + ' ' + str(line.split(' ')[3].rstrip('\n'))
                
    with open("equil1.nvt", 'w') as f:
        f.write(f"""parmfile solv.prmtop
coordinates solv.pdb
temperature {T_start}
timestep {timestep}
thermostat on
thermostatTemperature {T_start}
thermostatDamping 0.1
minimize {min_steps}
run {equil1_len}ns
restart {restart}
PME on
cutoff 9.0
switching on
switchDistance 7.5
atomRestraint "protein or resname LIG" setpoints 5@0
trajectoryFile equil1.dcd
trajectoryPeriod {dcdfreq}
boxSize {dimension}""")

    os.system("rm get_celldimension.vmd celldimension.log")
    os.system(f"acemd3 --device {device} equil1.nvt")



def equil2(restart):
    with open("equil2.npt", 'w') as f:
        f.write(f"""parmfile solv.prmtop
coordinates solv.pdb
binCoordinates output.coor
binVelocities output.vel
extendedSystem output.xsc
temperature {T_start}
timestep {timestep}
thermostat on
thermostatTemperature {T_start}
thermostatDamping 0.1
barostat on
barostatPressure 1.01325
run {equil2_len}ns
restart {restart}
PME on
cutoff 9.0
switching on
switchDistance 7.5
atomRestraint "protein and backbone or resname LIG" setpoints 5@0
trajectoryFile equil2.dcd
trajectoryPeriod {dcdfreq}""")

    os.system(f"acemd3 --device {device} equil2.npt")



def equil():
    os.chdir(f"{folder}/equil1")
    print('————————————————————\nRUNNING EQUILIBRATION\n————————————————————\n')
    if not os.path.exists(f"{folder}/equil1/equil1.dcd"):
        print('Running equil1')
        equil1('off')

    else:
        print('Check equil1 trajectory integrity')

        check_u = mda.Universe('solv.prmtop', 'equil1.dcd')
        check_frames = len(check_u.trajectory)
        equil1_frame = equil1_len / conversion_factor

        if float(check_frames) == float(equil1_frame):
            print('——Trajectory integrity checked')

        else:
            print('——Trajectory incomplete')
            
            if resume_check == True:
                print('————Resuming equil1')
                equil1('on')
                

            else:
                os.system(f'rm equil1.dcd')
                print('————Restarting equil1')
                equil1('off')

    os.system("cp output* ../equil2")

    os.chdir(f"{folder}/equil2")

    if not os.path.exists(f"{folder}/equil2/equil2.dcd"):
        print('Running equil2')
        equil2('off')

    else:
        print('Check equil2 trajectory integrity')

        check_u = mda.Universe('solv.prmtop', 'equil2.dcd')
        check_frames = len(check_u.trajectory)
        equil2_frame = equil2_len / conversion_factor

        if float(check_frames) == float(equil2_frame):
            print('——Trajectory integrity checked')

        else:
            print('——Trajectory incomplete')
            if resume_check == True:
                print('————Resuming equil2')
                equil2('on')

            else:
                print('————Restarting equil2')
                os.system(f'rm equil2.dcd')
                equil('off')

    os.system("cp output* ../MD")


def ref_fingerprint():
    global ref_fp
    if not os.path.exists(f'{folder}/MD/reference_ligand.pdb'):
        os.chdir(f"{folder}/equil2")
        ref_u = mda.Universe('solv.prmtop', 'equil2.dcd')

        ref = mda.Universe('solv.pdb') 
        protein = ref_u.select_atoms('protein')
        not_protein = ref_u.select_atoms('not protein')
        whole = ref_u.select_atoms('all')
        transforms = [trans.unwrap(whole),
                        trans.center_in_box(protein, wrap=False),
                        trans.wrap(whole, compound='residues')]
        ref_u.trajectory.add_transformations(*transforms)
        
        old_rmsd, new_rmsd = align.alignto(ref_u, ref, select='protein and backbone', weights='mass')
        
        ref_u.trajectory[-1]
        ref_u_ligand = ref_u.select_atoms('resname LIG')
        with mda.Writer('../MD/reference_ligand.pdb', ref_u_ligand.n_atoms) as W:
            W.write(ref_u_ligand)

        ref_u_protein = ref_u.select_atoms('protein')
        with mda.Writer('../MD/reference_protein.pdb', ref_u_protein.n_atoms) as W:
            W.write(ref_u_protein)
    
    os.chdir(f"{folder}/MD")
    protein = next(oddt.toolkit.readfile('pdb', 'reference_protein.pdb'))
    protein.protein = True
    ligand = next(oddt.toolkit.readfile('pdb', 'reference_ligand.pdb'))
    
    ref_fp = fingerprints.InteractionFingerprint(ligand, protein)

    os.chdir(f"{folder}/MD")



def run_temp(temperature, t_step, restart):
    with open("run.nvt", 'w') as f:
            f.write(f"""parmfile solv.prmtop
    coordinates solv.pdb
    temperature {temperature}
    binCoordinates output.coor
    binVelocities output.vel
    extendedSystem output.xsc
    timestep {timestep}
    thermostat on
    thermostatTemperature {temperature}
    thermostatDamping 0.1
    run {t_step}ns
    restart {restart}
    PME on
    cutoff 9.0
    switching on
    switchDistance 7.5
    trajectoryFile run_{temperature}.dcd
    trajectoryPeriod {dcdfreq}""")

    os.system(f"acemd3 --device {device} run.nvt")
    os.system(f"cp output.coor output_files/output_{temperature}.coor")
    os.system(f"cp output.vel output_files/output_{temperature}.vel")
    os.system(f"cp output.xsc output_files/output_{temperature}.xsc")




def avg_file(temperature):
    ifp_list = []
    with open(f'sim_{temperature}.dat', 'r') as f:
        lines = f.readlines()
        for line in lines:
            ifp_list.append(line.rstrip('\n'))
            
    sum_num = 0
    for x in ifp_list:
        sum_num = sum_num + float(x)           
    avg = sum_num / len(ifp_list)
    with open(f'avg_score', 'a') as avg_score:
        avg_score.writelines(str(avg) + "\n")



def titration(temperature, t_step):
    print(f'——Running {temperature} simulation')

    if not os.path.exists(f'run_{temperature}.dcd'):
        print(f'——run_{temperature}.dcd doesn\'t exists')

        run_temp(temperature, t_step, 'off')

    else:
        print(f'——run_{temperature}.dcd found')
        print('————Checking trajectory integrity')

        check_u = mda.Universe('solv.prmtop', f'run_{temperature}.dcd')
        check_frames = len(check_u.trajectory)
        run_frame = t_step / conversion_factor

        if check_frames == run_frame:
            print('————Trajectory integrity checked')

        else:
            if resume_check == True:
                print('——————Resuming trajectory')
                run_temp(temperature, t_step, 'on')

            else:
                print('——————Restarting trajectory')
                os.system(f'rm run_{temperature}.dcd')
                run_temp(temperature, t_step, 'off')


        
    trajectory = f'run_{temperature}.dcd'
    topology = 'solv.prmtop'
    
    wrap_trj = f"swag_{temperature}.dcd"
    
    if not os.path.exists(wrap_trj):
        print(f'——swag_{temperature}.dcd doesn\'t exists')
        blocks = trajectory_blocks(topology, trajectory)[0]
        wrapped_blocks = parallelizer.run(blocks, wrap_blocks, n_procs, 'Wrapping blocks')
        merge_trj(wrapped_blocks, wrap_trj, remove=True)

    else:
        print(f'——swag_{temperature}.dcd found')
        check_u = mda.Universe('solv.prmtop', f'swag_{temperature}.dcd')
        check_frames = len(check_u.trajectory)
        swag_frame = t_step / conversion_factor

        if check_frames == swag_frame:
            print('————Trajectory integrity checked')

        else:
            print('————Wrapped trj lenght mismatch\nRewrapping')
            blocks = trajectory_blocks(topology, trajectory)[0]
            wrapped_blocks = parallelizer.run(blocks, wrap_blocks, n_procs, 'Wrapping blocks')
            merge_trj(wrapped_blocks, wrap_trj, remove=True)



    if not os.path.exists(f'sim_{temperature}.dat'):
        print(f'——sim_{temperature}.dat doesn\'t exists')
        calculate_ifp(topology, wrap_trj, temperature)
    else:
        print(f'——sim_{temperature}.dat found')

    avg_file(temperature)
    
    if os.path.exists("frame_PDBs"):
        os.system("rm -r frame_PDBs")


def thermic_titration():

    os.chdir(f"{folder}/MD")
    if not os.path.exists('output_files'):
        os.mkdir('output_files')
    os.system(f"cp ../equil2/output.coor {folder}/MD/output.coor")
    os.system(f"cp ../equil2/output.vel {folder}/MD/output.vel")
    os.system(f"cp ../equil2/output.xsc {folder}/MD/output.xsc")

    ref_fingerprint()

    with open("avg_score", 'w') as avg:
        avg.write('')
    last_line = -1

    print('————————————————————\nRUNNING TTMD\n————————————————————\n')

    for set in temp_set:
        t_a = set[0]
        t_z = set[1]
        interval = set[2]
        t_step = set[3]
        for temperature in range(t_a, t_z + interval, interval):
            if last_line != int(0) or last_line != float(0):
                print(f'\nTEMPERATURE {temperature}\n')
                titration(temperature, t_step)
                with open("avg_score", 'r') as avg:
                    line = avg.readlines()
                    last_line = line[-1]
                done_temperature.append(temperature)

    merge_sim()
    
    return done_temperature



def final_merge_trj():
    os.chdir(f'{folder}/MD')
    
    if not os.path.exists('merged_swag.dcd'):
        trj_list = []
        for t in done_temperature:
            trj = f'swag_{t}.dcd'
            trj_list.append(trj)
        
        merge_trj(trj_list, 'merged_swag.dcd', remove=False)
        os.listdir(os.getcwd())
    
    if dryer == 'yes':
        blocks = trajectory_blocks('solv.pdb', 'merged_swag.dcd')[0]
        dryed_blocks = parallelizer.run(blocks, dry_trj, n_procs, 'Drying blocks')
        merge_trj(dryed_blocks, 'dry.dcd', remove=True)
        
        u = mda.Universe('solv.pdb')
        dry_sel = u.select_atoms("not resname WAT and not resname Na+ and not resname Cl-")
        with mda.Writer('dry.pdb', dry_sel.n_atoms) as W:
            W.write(dry_sel)
            
        os.system('rm merged_swag.dcd')



def merge_sim():
    os.chdir(f'{folder}/MD')

    sim_list = [-1]
    for t in done_temperature:
        sim = f'sim_{t}.dat'
        with open(sim, 'r') as sim:
            lines = sim.readlines()
            for line in lines:
                sim_list.append(line.rstrip('\n'))

    if os.path.exists('similarity.dat'):
        os.system('rm similarity.dat')
        
    with open('similarity.dat', 'a') as f:
            for i in sim_list:
                f.write(f'{i }\n')

    return sim_list



def sim_list():
    os.chdir(f'{folder}/MD')
    sim_list = []
    lines = []

    if not os.path.exists('similarity.dat'):
        sim_list = merge_sim()
    else:
        with open('similarity.dat', 'r') as f:
            lines = f.readlines()

        for i in lines:
            sim = i.rstrip('\n')
            sim_list.append(sim)

    return sim_list

def getSim():

    print("\nRetrieving similarity values...")
    sim_list = []
    with open(f'{folder}/MD/similarity.dat', 'r') as f:
        lines = f.readlines()
        for i in lines:
            sim = i.rstrip('\n')
            sim_list.append(float(sim))

    if smooth != 0:
        smooth_sim = []
        for n,i in enumerate(sim_list):
            sum = i
            count = 1
            for x in range (1, smooth+1):
                minor_smooth = n-x
                major_smooth = n+x
                
                if minor_smooth >= 0:
                    sum += sim_list[minor_smooth]
                    count += 1

                if major_smooth < len(sim_list):
                    sum += sim_list[major_smooth]
                    count += 1

            mean = sum / count
            smooth_sim.append(mean)
        
        return smooth_sim

    else:
        return sim_list



def getTime():
    os.chdir(f'{folder}/MD')
    total_ns = 0
    for set in temp_set:
        t_a = set[0]
        t_z = set[1]
        interval = set[2]
        t_step = set[3]
        for temperature in range(t_a, t_z + interval, interval):
            if os.path.exists(f'sim_{temperature}.dat'):
                done_temperature.append(temperature)
                total_ns += t_step
    
    total_ts = int(total_ns / conversion_factor)
    time_list = []
    for i in range(0, total_ts + 1):
        time = i * conversion_factor
        time_list.append(time)

    return time_list



def calcRMSD():
    if dryer == 'yes':
        topology = 'dry.pdb'
        trajectory = 'dry.dcd'
    else:
        topology = 'solv.pdb'
        trajectory = 'merged_swag.dcd'

    u = mda.Universe(topology, trajectory)
    R = rms.RMSD(u, u, select='backbone', groupselections=['resname LIG'], ref_frame=0).run()
    rmsd_backbone = R.rmsd.T[2]
    rmsd_ligand = R.rmsd.T[3]
    return rmsd_backbone, rmsd_ligand



def titration_timeline():
    os.chdir(f'{folder}/MD')
    #### this function plots IFPcs and both backbone and ligand RMSD vs simulation time
    print("\nPlotting titration timeline...")

    time_list = getTime()
    sim_list = getSim()
    rmsd_list = calcRMSD()

    #create temperature list
    temperature_list = [T_start]
    for set in temp_set:
        t_a = set[0]
        t_z = set[1]
        interval = set[2]
        t_step = set[3]
        for temperature in range(t_a, t_z + interval, interval):
            frames_per_step = int(t_step/conversion_factor)
            for i in range(frames_per_step):
                temperature_list.append(temperature)

    def add_colorbar_outside(im,ax):
        fig = ax.get_figure()
        bbox = ax.get_position() #bbox contains the [x0 (left), y0 (bottom), x1 (right), y1 (top)] of the axis.
        height = 0.4
        width = 0.01
        eps = 0.02 #margin between plot and colorbar
        pad = 0.03
        # [left most position, bottom position, width, height] of color bar.
        cax = fig.add_axes([bbox.x1 + eps, bbox.y0 + pad, width, height])#bbox.height])
        cbar = fig.colorbar(im, cax=cax, ticks=done_temperature)
        return cbar



    # plot IFPcs
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,9))
    x = time_list
    y = np.array(sim_list).astype(float)
    divnorm = mplcolors.TwoSlopeNorm(vmin=T_start, vcenter=(T_start+T_stop)/2, vmax=T_stop)
    axs[0].set_title('Interaction Fingerprint Similarity')
    axs[0].set_xlabel('Time (ns)')
    axs[0].set_ylabel('IFP$_{CS}$')
    axs[0].set_ylim(-1,0)
    x_lim = ( temp_set[0][1] - temp_set[0][0] ) / temp_set[0][2] * temp_set[0][3] + temp_set[0][3]
    axs[0].set_xlim(0,x_lim)
    s = axs[0].scatter(x,y, c=temperature_list, cmap='RdYlBu_r', norm=divnorm)
    cbar = add_colorbar_outside(s, ax=axs[0])
    cbar.set_label('Temperature (K)', rotation=270, labelpad=15)
    cbar.ax.set_yticklabels(np.array(done_temperature).astype('str'))

    # plot RMSD
    x1 = time_list[1:]
    y1 = rmsd_list[0]
    xnew1 = np.linspace(x1[0], x1[-1], smooth) 
    spl1 = make_interp_spline(x1, y1, k=5)
    power_smooth1 = spl1(xnew1)
    axs[1].plot(xnew1, power_smooth1, color='seagreen', label='Backbone')
    x2 = time_list[1:]
    y2 = rmsd_list[1]
    xnew2 = np.linspace(x2[0], x2[-1], smooth) 
    spl2 = make_interp_spline(x2, y2, k=5)
    power_smooth2 = spl2(xnew2)
    axs[1].plot(xnew2, power_smooth2, color='tomato', label='Ligand')
    axs[1].set_ylabel('RMSD (Å)')
    axs[1].set_xlabel('Time (ns)')
    axs[1].set_title('RMSD')
    axs[1].set_xlim(0,x_lim)
    axs[1].set_ylim(0)
    axs[1].legend()
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.draw_without_rendering() #temporary fix for bug with colorbar label in matplotlb version 3.5.1
    fig.savefig('titration_timeline.png', dpi=300)



def titration_profile():
    os.chdir(f'{folder}/MD')
    print("\nPlotting titration profile...")
    #### this function plots average IFPcs vs temperature
    done_temperature = []
    for temp in range(temp_set[0][0],temp_set[-1][1]+temp_set[0][2], temp_set[0][2]):
        done_temperature.append(temp)
    temperature_array = np.array(done_temperature).astype(int)
    fig, axs = plt.subplots(nrows=1, ncols=1)
    avg_list = []
    with open('avg_score', 'r') as avg:
        lines = avg.readlines()
        for line in lines:
            avg_list.append(float(line.rstrip('\n')))
    first_last_T = [T_start, T_stop]
    axs.set_xlim(first_last_T)
    axs.set_ylim(-1,0)
    axs.scatter(temperature_array, avg_list, c='royalblue')
    first_last_score = [-1.0, avg_list[-1]]
    f = np.poly1d(np.polyfit(first_last_T, first_last_score, 1))
    slope, intercept, r_value, p_value, std_err = linregress(first_last_T, first_last_score)
    axs.plot(temperature_array, f(temperature_array), color='tomato', ls='--', label="MS = {:.5f}".format(slope))
    axs.set_title('Titration Profile')
    axs.set_xlabel('Temperature (K)')
    axs.set_ylabel('Average IFP$_{CS}$')
    axs.set_ylim(-1,0)
    axs.set_xlim(first_last_T)
    axs.legend()
    fig.savefig('titration_profile.png', dpi=300)



print(header)
MAIN()
