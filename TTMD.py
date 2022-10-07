#!/usr/bin/env python

### comment MAIN row(s) to skip
def MAIN():
    ### PREPARATORY STEPS 
    prepare_system()
    statistics()
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
  # if launched with nohup: in case of need, kill all child processes
  # executing kill.py script, created after starting main program
# N.B. this script works with python3

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
import argparse
import configparser
import glob
from tabulate import tabulate
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mplcolors
import oddt
from oddt import fingerprints
import MDAnalysis as mda
import MDAnalysis.transformations as trans
from MDAnalysis.analysis import align
from MDAnalysis.analysis import rms
import sklearn.metrics
from scipy.interpolate import make_interp_spline
from scipy.stats import linregress
import multiprocessing
import tqdm

np.set_printoptions(threshold=sys.maxsize)
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

script_name = os.path.basename(__file__)

folder = os.getcwd()

### components selection for statistics
components = ['resname WAT', 'protein', 'resname LIG', 'resname Na+', 'resname Cl-']

done_temperature = []

ref_fp = ''

###############################################################################
### PARSE CONFIG

def parse_input():
    # initialize parser object
    parser = argparse.ArgumentParser(description='Thermal Titration Molecular Dynamics')

    ## required arguments are defined as optional and manually checked afterwards
    # required arguments
    parser.add_argument("-f", "--config_file", type=str, help='Config file (overrides command line options)', metavar='', dest='config_file')
    parser.add_argument('-p', '--protein_name', help='Protein file in the .pdb format [REQUIRED]', metavar='', dest='protein_name')
    parser.add_argument('-l', '--ligand_name', help='Ligand file in the .mol2 format [REQUIRED]', metavar='', dest='ligand_name')
    parser.add_argument('-fc', '--ligand_charge', type=int, help='Ligand formal charge [REQUIRED]', metavar='', dest='ligand_charge')
    parser.add_argument('-vp', '--vmd_path', help='path/to/vmd_executable [REQUIRED]', metavar='', dest='vmd')
    # optional arguments
    parser.add_argument('-pd', '--padding', default=15, type=int, help='Padding value for simulation box (Å), default=15', metavar='', dest='padding')
    parser.add_argument('-i', '--iso', default='no', type=str, help='Flag to build cubic box (bool), default=no', metavar='', dest='iso')
    parser.add_argument('-tr', '--temp_ramp', default=[[300, 450, 10, 10],], type=list, help='Temperature ramp (list), default=[[300, 450, 10, 10]]', metavar='', dest='temp_ramp')
    parser.add_argument('-ts', '--timestep', default=2, type=int, help='Timestep (fs) for MD simulations, default=2', metavar='', dest='timestep')
    parser.add_argument('-df', '--dcdfreq', default=10000, type=int, help='Period of the trajectory files, default=10000', metavar='', dest='dcdfreq')
    parser.add_argument('-ms', '--min_steps', default=500, type=int, help='Minimization steps with the cg algorithm before equilibration, default=500', metavar='', dest='min_steps')
    parser.add_argument('-e1', '--equil1_len', default=0.1, type=float, help='Lenght of the first (NVT) equilibration stage (ns), default=0.1', metavar='', dest='equil1_len')
    parser.add_argument('-e2', '--equil2_len', default=0.5, type=float, help='Lenght of the second (NPT) equilibration stage (ns), default=0.5', metavar='', dest='equil2_len')
    parser.add_argument('-r', '--resume', default='yes', type=str, help='Resume simulations or restart from the beginning of the step, default=yes', metavar='', dest='resume')
    parser.add_argument('-st', '--stride', default=1, type=int, help='Stride for the final (merged) trajectory, default=1', metavar='', dest='stride')
    parser.add_argument('-d', '--dryer', default='yes', type=str, help='Remove water and ions from output trajectory, default=yes', metavar='', dest='dryer')
    parser.add_argument('-sm', '--smooth', default=200, type=int, help='Smoothing for the curve reported on output plots, default=200', metavar='', dest='smooth')
    parser.add_argument('-dv', '--device', default=0, type=int, help='Index of GPU device to use for MD simulations, default=0', metavar='', dest='device')
    parser.add_argument('-np', '--n_procs', default=4, type=int, help='Number of CPU cores to use for trajectory analysis, default=4', metavar='', dest='n_procs')
    
    args = parser.parse_args()

    global config_name
    config_name = args.config_file

    # if config file is provided, options are read directly from it and used to replace the default values
    if args.config_file:
        config = configparser.ConfigParser()
        config.read(args.config_file)
        defaults = {}
        defaults.update(dict(config.items("Defaults")))
        # configparser cannot properly read lists from config files
        if 'temp_ramp' in defaults.keys():
            import ast
            my_list = ast.literal_eval(config.get("Defaults", "temp_ramp"))
            defaults['temp_ramp'] = my_list    
        parser.set_defaults(**defaults)
        args = parser.parse_args() # Overwrite arguments

    # setup variables for script execution from user-defined parameters

    #check existence and correct format of protein file
    try:
        global protein_name
        protein_name = os.path.abspath(args.protein_name)
    except Exception:
        print('Protein path missing! (check your config file)')
        sys.exit(0)
    if not os.path.exists(protein_name):
        print(f'{protein_name} is not a valid path')
        sys.exit(0)
    elif protein_name[-3:] != 'pdb':
        print('Protein must be in pdb format')
        sys.exit(0)

    #check existence and correct format of protein file
    try:
        global ligand_name
        ligand_name = os.path.abspath(args.ligand_name)
    except Exception:
        print('Ligand path missing! (check your config file)')
        sys.exit(0)
    if not os.path.isfile(ligand_name):
        print(f'{ligand_name} is not a valid path')
        sys.exit(0)
    elif ligand_name[-4:] != 'mol2':
        print('Ligand must be in mol2 format')
        sys.exit(0)

    #check existence of ligand charge
    global ligand_charge
    ligand_charge = args.ligand_charge
    if ligand_charge == None:
        print('Ligand charge missing! (check your config file)')
        sys.exit(0)

    global padding
    padding = args.padding
    global iso
    if args.iso == 'yes':
        iso = 'iso'
    elif args.iso == 'no':
        iso = ''
    else:
        sys.exit('invalid iso settings')

    #check correct construction of the temperature ramp list
    global temp_set
    temp_set = args.temp_ramp
    ramp_check = True
    count = 0
    temp_list = []
    for sublist in temp_set:
        count += 1
        #check if the temperature step is correctly set
        t_start = sublist[0]
        t_end = sublist[1]
        T_step = sublist[2]
        if (t_end-t_start) % T_step != 0:
            ramp_check = False
            print('\nTemperature ramp is not set up correctly!')
            print(f'--> List n° {count} contains an invalid temperature step ({T_step})\n')
        #check if each list has the right number of elements
        num_el = len(sublist)
        if num_el != 4:
            ramp_check = False
            print('\nTemperature ramp is not set up correctly!')
            print(f'--> List n° {count} contains only {num_el} elements!\n')
    #if one condition is not satisfied, exit the program
    if not ramp_check:
        print(f'\nYour ramp: {temp_set}\nThe right way: [[T_start (K), T_end (K), T_step (K), step_len (ns)],]\n')
        sys.exit(0)

    global T_start
    T_start = temp_set[0][0]
    global T_stop
    T_stop = temp_set[-1][1]

    global timestep
    timestep = args.timestep
    global dcdfreq
    dcdfreq = args.dcdfreq
    global min_steps
    min_steps  = args.min_steps
    global equil1_len
    equil1_len = args.equil1_len
    global equil2_len
    equil2_len = args.equil2_len
    global resume
    if args.resume == 'yes':
        resume = True
    elif args.resume == 'no':
        resume = False
    else:
        sys.exit('invalid resume settings')
    global stride
    stride = args.stride

    global conversion_factor
    conversion_factor = timestep * dcdfreq * stride / 1000000

    global dryer
    if args.dryer:
        dryer = 'yes'
    else:
        dryer = ''
    global smooth
    smooth = args.smooth
    global device
    device = args.device
    global n_procs
    n_procs = args.n_procs
    global vmd
    #check if provided vmd path is correct: if not, search for local installation of vmd and use that instead
    vmd_check = True
    #control first if vmd path is provided
    try:
        vmd = os.path.abspath(args.vmd_path)
    except Exception:
        print('\nVMD path missing! (check your config file)\n')
        vmd_check = False
    #control if provided path is a valid path
    if vmd_check and not os.path.isfile(vmd):
        print(f'\n{vmd} is not a valid path\n')
        vmd_check = False
    #control if provided vmd path refers to a vmd installation
    if vmd_check and os.path.isfile(vmd):
        exe = vmd.split('/')[-1]
        if 'vmd' not in exe:
            print(f'\n{vmd} is not a valid VMD executable\n')
            vmd_check = False
    if not vmd_check:
        #if vmd is not installed on local machine, exit from the program
        import subprocess
        try:
            vmd_installed_path = str(subprocess.check_output(['which','vmd']))[2:-3]
        except Exception:
            print('\nVMD is not installed on your machine!\n')
            sys.exit(0)
        print(f'\nFound existing installation of VMD at {vmd_installed_path}')
        print(f'Using {vmd_installed_path}\n')
        vmd = vmd_installed_path

    # write config file with user-defined parameters for reproducibility reason
    vars_file = f'''
    [Defaults]

    #system preparation
    protein_name = {protein_name}
    ligand_name = {ligand_name}
    ligand_charge = {ligand_charge}
    padding = {padding}
    iso = {args.iso}

    #simulation setup
    temp_ramp = {temp_set}
    timestep = {timestep}
    dcdfreq = {dcdfreq}
    min_steps  = {min_steps}
    equil1_len = {equil1_len}
    equil2_len = {equil2_len}
    resume = {resume}

    #postprocessing & analysis
    stride = {stride}
    dryer = {args.dryer}
    smooth = {smooth}

    #hardware settings
    device = {device}
    n_procs = {n_procs}

    #external dependencies
    vmd = {vmd}
    '''

    with open('vars.dat','w') as f:
        f.write(vars_file)

    # print settings used for the current ttmd run, as stored in the 'vars.dat' file
    print('\n** Parameters for your simulations were stored in vars.dat **\n')
    print('\n#######################################################\n')
    print(vars_file)
    print('\n#######################################################\n')
    
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
            if f == os.path.basename(ligand_name) or f == os.path.basename(protein_name) or f == script_name or f == config_name or f == 'vars.dat':
                continue
            else:
                os.system(f'rm -r {f}')
                print(f'{f} removed')

        with open(f'{folder}/gpu.info', 'w') as i:
            i.write(gpu_check().rstrip('/n'))

###############################################################################

def pid():
    main_pid = os.getpid()
    with open('main_pid.log', 'w') as p:
        p.write(str(main_pid))

    with open('kill.py', 'w') as kill:
        kill.write(f'''
import os

with open('main_pid.log', 'r') as main:
    main_pid = int(main.readlines()[0])

os.system(f'pgrep -g {{main_pid}}> pid.log')

with open('pid.log', 'r') as log:
    lines = log.readlines()
    for line in lines:
        pid = int(line.rstrip('\\n'))
        os.system(f'kill {{pid}}')
''')

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



if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")

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
        
        if __name__ == '__main__':
            for i in range(n_procs):
                multiprocessing.Process(target=worker, args=(task_queue, done_queue)).start()
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

def wrapping(topology, trajectory, wrap_trj):
    blocks = trajectory_blocks(topology, trajectory)[0]
    wrapped_blocks = parallelizer.run(blocks, wrap_blocks, n_procs, 'Wrapping blocks')
    merge_trj(topology, wrapped_blocks, wrap_trj, remove=True)



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
    ligand = u.select_atoms('resname LIG')
#    transforms = [trans.unwrap(u.atoms),
#                trans.center_in_box(ligand, center='geometry'),
#                trans.wrap(u.atoms, compound='residues'),
#                trans.unwrap(u.atoms),
#                trans.center_in_box(protein, center='geometry'),
#                trans.wrap(not_protein, compound='residues')]
    transforms = [trans.unwrap(u.atoms),
                trans.center_in_box(protein, center='geometry'),
                trans.wrap(not_protein, compound='residues')]

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



def merge_trj(topology, trj_list, trj_name, remove=bool):
    u = mda.Universe(topology, trj_list)

    with mda.Writer(trj_name, u.atoms.n_atoms) as W:
        for ts in u.trajectory:
            W.write(u.atoms)
    
    if remove == True:
        for trj in trj_list:
            os.system(f'rm {trj}')



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
    ref_fp = list(args)[2]

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
    global ref_fp
    u = mda.Universe(topology, trajectory)
    ts_list = []
    for ts in u.trajectory:
        ts_list.append([(ts.frame,), (topology,), (trajectory,)])
    
    if not os.path.exists("frame_PDBs"):
        os.mkdir("frame_PDBs")

    pdb_list = parallelizer.run(ts_list, write_frame, n_procs, 'Writing PDBs')

    calc_ifp = []
    for i in pdb_list:
        el = []
        for x in i:
            el.append(x)
        el.append(ref_fp)
        calc_ifp.append(tuple(el))
            
    ifp_list = parallelizer.run(calc_ifp, ifp, n_procs, 'Calculating fingerprints')
    
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



def prepare_system():
    os.chdir(folder)

    if not os.path.exists('solv.pdb') and not os.path.exists('solv.prmtop'):
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
    solvatebox COMPL TIP3PBOX {padding} {iso}
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
    solvatebox COMPL TIP3PBOX {padding} {iso}
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

        statistics()



def trj_check_start(top, trj, trj_len):

    checked = False

    check_u = mda.Universe(top, trj)
    check_frames = len(check_u.trajectory)
    req_frames = trj_len / conversion_factor

    if float(check_frames) == float(req_frames):
        print('——Trajectory integrity checked')
        checked = True
    
    return checked
    


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

    if trj_check_start('solv.pdb', 'equil1.dcd', equil1_len) == False:
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

        if trj_check_start('solv.pdb', 'equil2.dcd', equil2_len) == False:

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
    if not os.path.exists(f'{folder}/MD/reference_ligand.pdb'):
        ref_u = mda.Universe(f'{folder}/equil2/solv.prmtop', f'{folder}/equil2/equil2.dcd')

        ref = mda.Universe(f'{folder}/equil2/solv.pdb') 
        protein = ref_u.select_atoms('protein')
        not_protein = ref_u.select_atoms('not protein')
        whole = ref_u.select_atoms('all')
        ligand = ref_u.select_atoms('resname LIG')
        transforms = [trans.unwrap(ref_u.atoms),
            trans.center_in_box(ligand, center='geometry'),
            trans.wrap(ref_u.atoms, compound='residues'),
            trans.unwrap(ref_u.atoms),
            trans.center_in_box(protein, center='geometry'),
            trans.wrap(not_protein, compound='residues')]

        ref_u.trajectory.add_transformations(*transforms)
        
        old_rmsd, new_rmsd = align.alignto(ref_u, ref, select='protein and backbone', weights='mass')
        
        ref_u.trajectory[-1]
        ref_u_ligand = ref_u.select_atoms('resname LIG')
        with mda.Writer(f'{folder}/MD/reference_ligand.pdb', ref_u_ligand.n_atoms) as W:
            W.write(ref_u_ligand)

        ref_u_protein = ref_u.select_atoms('protein')
        with mda.Writer(f'{folder}/MD/reference_protein.pdb', ref_u_protein.n_atoms) as W:
            W.write(ref_u_protein)

        #### create ttmd.pdb, input for production simulations
        last_step = ref_u.select_atoms('all')
        with mda.Writer(f'{folder}/MD/solv.pdb', last_step.n_atoms) as W:
            W.write(last_step)        
    
    protein = next(oddt.toolkit.readfile('pdb', f'{folder}/MD/reference_protein.pdb'))
    protein.protein = True
    ligand = next(oddt.toolkit.readfile('pdb', f'{folder}/MD/reference_ligand.pdb'))
    
    ref_fp = fingerprints.InteractionFingerprint(ligand, protein)

    return ref_fp



def run_temp(temperature, t_step, restart):
    #### first step starts from pdb (last frame of equil2), prmtop and xsc, as in sumd 
    if temperature == T_start:
        with open("run.nvt", 'w') as f:
                f.write(f"""parmfile solv.prmtop
        coordinates solv.pdb
        temperature {temperature}
        #binCoordinates output.coor
        #binVelocities output.vel
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
    else:
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
        
        if trj_check_start('solv.pdb', f'run_{temperature}.dcd', t_step) == False:

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
        wrapping(topology, trajectory, wrap_trj)

    elif trj_check_start('solv.pdb', f'run_{temperature}.dcd', t_step) == False:

        print('————Wrapped trj lenght mismatch\nRewrapping')
        wrapping(topology, trajectory, wrap_trj)


    global ref_fp
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

    global ref_fp
    ref_fp = ref_fingerprint()

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
                    last_line = float(line[-1])
                done_temperature.append(temperature)

    merge_sim()
    
    return done_temperature



def final_merge_trj():
    os.chdir(f'{folder}/MD')
    
    if not os.path.exists('merged_swag.dcd'):
        topology = 'solv.prmtop'
        trj_list = []
        for t in done_temperature:
            trj = f'swag_{t}.dcd'
            trj_list.append(trj)
        
        merge_trj(topology, trj_list, 'merged_swag.dcd', remove=False)
        os.listdir(os.getcwd())
    
    if dryer == 'yes':
        topology = 'dry.pdb'
        u = mda.Universe('solv.pdb')
        dry_sel = u.select_atoms("not resname WAT and not resname Na+ and not resname Cl-")
        with mda.Writer('dry.pdb', dry_sel.n_atoms) as W:
            W.write(dry_sel)
        
        blocks = trajectory_blocks('solv.pdb', 'merged_swag.dcd')[0]
        dryed_blocks = parallelizer.run(blocks, dry_trj, n_procs, 'Drying blocks')
        merge_trj(topology, dryed_blocks, 'dry.dcd', remove=True)
        
        
            
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
            if os.path.exists(f'sim_{temperature}.dat'):
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
    temperature_list = []
    for set in temp_set:
        t_a = set[0]
        t_z = set[1]
        interval = set[2]
        t_step = set[3]
        for temperature in range(t_a, t_z + interval, interval):
            if os.path.exists(f'sim_{temperature}.dat'):
                temperature_list.append(temperature)

    temperature_array = np.array(temperature_list).astype(int)
    fig, axs = plt.subplots(nrows=1, ncols=1)
    avg_list = []
    with open('avg_score', 'r') as avg:
        lines = avg.readlines()
        for line in lines:
            avg_list.append(float(line.rstrip('\n')))
    first_last_T = [temperature_list[0], temperature_list[-1]]
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


if __name__ == '__main__':
    print(header)
    parse_input()
    pid()
    gpu_info()
    MAIN()
