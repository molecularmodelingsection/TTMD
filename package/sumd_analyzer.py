#### set topology (pdb) and trajectory (dcd) files
topology = 'dry.pdb'
trajectory = 'stride_10.dcd'

#### if using Amber, itopology e parameters are the same files
itopology = 'complex.prmtop'
parameters = 'complex.prmtop'

#### receptor and ligand selection (vmd style)
rec_parameters = 'protein.prmtop'
receptorSel = 'protein'

lig_parameters = 'nucleic.prmtop'
ligandSel = 'nucleic'


#### resids for binding site definition (the same used in the input file for SuMD simulation)
receptorResids = '28 54 85 88 104 105 106 107 108 120 121 123'
ligandResids = '134 136 137 146 147 154 156 157 158'

#### cutoff distance for receptor-ligand contacts calculation
distanceCutoff = 4.5

#### integration timestep for molecular dynamics simulations
timestep = 2
####
dcdfreq = 10000
#### timestep interval between each trajectory frame
stride = 10

#### number of receptor/ligand residues to consider for the analysis (the most contacted ones)
numResidRec = 25
numResidLig = 25

#### residue number correction (useful if working with tleap/AMBER)
numShiftRec = +3  ### number to add to resid number to align tleap to fasta
numShiftLig = -130 ### number to add to resid number to align tleap to fasta

#### path to NAMD executable
namdPATH = '/home/smenin/Programs/NAMD_2.14/namd2'

#### number of processors for interaction energy calculations
n_procs = 5

video = True

#### if True, calculate ligand RMSD vs reference structure. If False, calculate dcm binding site to ligand
ref_bool = True

#### reference pdb structure for RMSD calculation (only necessary if ref_bool = True
reference_pdb = 'reference.pdb'

theme = 'light'
transparent = False
color_palette = 'boh'
font = ''
fontsize = 14
 
            

'''
████████████████████████████████████████████████████████████████████████████████████████████████████
'''

header = '''

            ███████ ██    ██ ███    ███ ██████                                 
            ██      ██    ██ ████  ████ ██   ██                                
            ███████ ██    ██ ██ ████ ██ ██   ██                                
                 ██ ██    ██ ██  ██  ██ ██   ██                                
            ███████  ██████  ██      ██ ██████                                 
                                                                   
                                                                   
 █████  ███    ██  █████  ██      ██    ██ ███████ ███████ ██████  
██   ██ ████   ██ ██   ██ ██       ██  ██     ███  ██      ██   ██ 
███████ ██ ██  ██ ███████ ██        ████     ███   █████   ██████  
██   ██ ██  ██ ██ ██   ██ ██         ██     ███    ██      ██   ██ 
██   ██ ██   ████ ██   ██ ███████    ██    ███████ ███████ ██   ██ 


                    M.Pavan 13/01/2022
'''
help = '''
\nHow to run: python3 RNASuMDAnalyzer.py [PROTOCOL]
Available protocols:
    -geometry
    -mmgbsa
    -intEnergy
    -perResRec
    -perResLig
    -matrix

    -video
'''

'''
████████████████████████████████████████████████████████████████████████████████████████████████████
'''

import os
import sys
import argparse
import configparser
import glob
from statistics import mean
from collections import Counter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mplcolors
import seaborn as sns
import MDAnalysis as mda
import MDAnalysis.analysis.rms
import MDAnalysis.analysis.align as align
from scipy.stats import iqr
from scipy.interpolate import make_interp_spline, BSpline
import barnaba as bb
import multiprocessing
import tqdm
from termcolor import colored

##################################################

cf = (timestep * dcdfreq * stride) / (10**6)

_wd = os.path.dirname(__file__)

def wd(var):
    path = f'{_wd}/{var}'
    return path

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")

top = wd(topology)
trj = wd(trajectory)
itop = wd(itopology)
par = wd(parameters)
ref_pdb = wd(reference_pdb)
rec_parm = wd(rec_parameters)
lig_parm = wd(lig_parameters)



if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
np.set_printoptions(threshold=sys.maxsize)

'''
████████████████████████████████████████████████████████████████████████████████████████████████████
'''


def parse_input():
    # initialize parser object
    parser = argparse.ArgumentParser(description='SuMD Analyzer')

    ## required arguments are defined as optional and manually checked afterwards
    # required arguments
    parser.add_argument("-f", "--config_file", type=str, help='Config file (overrides command line options)', metavar='', dest='config_file')
    parser.add_argument('-top', '--topology', type=str, help='MD simulation Topology (.pdb)[REQUIRED]', metavar='', dest='topology')
    parser.add_argument('-trj', '--trajectory', type=str, help='MD simulation Trajectory (.dcd)[REQUIRED]', metavar='', dest='trajectory')
    parser.add_argument('-itop', '--itopology', type=str, help='Complex topology (.prmtop/.psf)[REQUIRED]', metavar='', dest='itopology')
    parser.add_argument('-par', '--parameters', type=str, help='Complex topology (.prmtop/.prm)[Required only for Charmm MD]', metavar='', dest='par')
    parser.add_argument('-rec', '--receptor_par', type=str, help='Receptor parametrs (.prmtop)', metavar='', dest='rec_par')
    parser.add_argument('-rec', '--receptor_sel', type=str, help='Receptor selection (vmd style)', metavar='', dest='rec_sel')
    # parser.add_argument('-rpar', '--receptor_par', help='path/to/vmd_executable [REQUIRED]', metavar='', dest='vmd')
    # # optional arguments
    # parser.add_argument('-pd', '--padding', default=15, type=int, help='Padding value for simulation box (Å), default=15', metavar='', dest='padding')
    # parser.add_argument('-i', '--iso', default='no', type=str, help='Flag to build cubic box (bool), default=no', metavar='', dest='iso')
    # parser.add_argument('-tr', '--temp_ramp', default=[[300, 450, 10, 10],], type=list, help='Temperature ramp (list), default=[[300, 450, 10, 10]]', metavar='', dest='temp_ramp')
    # parser.add_argument('-ts', '--timestep', default=2, type=int, help='Timestep (fs) for MD simulations, default=2', metavar='', dest='timestep')
    # parser.add_argument('-df', '--dcdfreq', default=10000, type=int, help='Period of the trajectory files, default=10000', metavar='', dest='dcdfreq')
    # parser.add_argument('-ms', '--min_steps', default=500, type=int, help='Minimization steps with the cg algorithm before equilibration, default=500', metavar='', dest='min_steps')
    # parser.add_argument('-e1', '--equil1_len', default=0.1, type=float, help='Lenght of the first (NVT) equilibration stage (ns), default=0.1', metavar='', dest='equil1_len')
    # parser.add_argument('-e2', '--equil2_len', default=0.5, type=float, help='Lenght of the second (NPT) equilibration stage (ns), default=0.5', metavar='', dest='equil2_len')
    # parser.add_argument('-r', '--resume', default='yes', type=str, help='Resume simulations or restart from the beginning of the step, default=yes', metavar='', dest='resume')
    # parser.add_argument('-st', '--stride', default=1, type=int, help='Stride for the final (merged) trajectory, default=1', metavar='', dest='stride')
    # parser.add_argument('-d', '--dryer', default='yes', type=str, help='Remove water and ions from output trajectory, default=yes', metavar='', dest='dryer')
    # parser.add_argument('-sm', '--smooth', default=200, type=int, help='Smoothing for the curve reported on output plots, default=200', metavar='', dest='smooth')
    # parser.add_argument('-dv', '--device', default=0, type=int, help='Index of GPU device to use for MD simulations, default=0', metavar='', dest='device')
    # parser.add_argument('-np', '--n_procs', default=4, type=int, help='Number of CPU cores to use for trajectory analysis, default=4', metavar='', dest='n_procs')
    
    # args = parser.parse_args()



    # global config_name
    # config_name = args.config_file

    # # if config file is provided, options are read directly from it and used to replace the default values
    # if args.config_file:
    #     config = configparser.ConfigParser()
    #     config.read(args.config_file)
    #     defaults = {}
    #     defaults.update(dict(config.items("Defaults")))
    #     # configparser cannot properly read lists from config files
    #     if 'temp_ramp' in defaults.keys():
    #         import ast
    #         my_list = ast.literal_eval(config.get("Defaults", "temp_ramp"))
    #         defaults['temp_ramp'] = my_list    
    #     parser.set_defaults(**defaults)
    #     args = parser.parse_args() # Overwrite arguments

    # # setup variables for script execution from user-defined parameters

    # #check existence and correct format of protein file
    # try:
    #     global protein_name
    #     protein_name = os.path.abspath(args.protein_name)
    # except Exception:
    #     print('Protein path missing! (check your config file)')
    #     sys.exit(0)
    # if not os.path.exists(protein_name):
    #     print(f'{protein_name} is not a valid path')
    #     sys.exit(0)
    # elif protein_name[-3:] != 'pdb':
    #     print('Protein must be in pdb format')
    #     sys.exit(0)

    # #check existence and correct format of protein file
    # try:
    #     global ligand_name
    #     ligand_name = os.path.abspath(args.ligand_name)
    # except Exception:
    #     print('Ligand path missing! (check your config file)')
    #     sys.exit(0)
    # if not os.path.isfile(ligand_name):
    #     print(f'{ligand_name} is not a valid path')
    #     sys.exit(0)
    # elif ligand_name[-4:] != 'mol2':
    #     print('Ligand must be in mol2 format')
    #     sys.exit(0)

    # #check existence of ligand charge
    # global ligand_charge
    # ligand_charge = args.ligand_charge
    # if ligand_charge == None:
    #     print('Ligand charge missing! (check your config file)')
    #     sys.exit(0)

    # global padding
    # padding = args.padding
    # global iso
    # if args.iso == 'yes':
    #     iso = 'iso'
    # elif args.iso == 'no':
    #     iso = ''
    # else:
    #     sys.exit('invalid iso settings')

    # #check correct construction of the temperature ramp list
    # global temp_set
    # temp_set = args.temp_ramp
    # ramp_check = True
    # count = 0
    # temp_list = []
    # for sublist in temp_set:
    #     count += 1
    #     #check if the temperature step is correctly set
    #     t_start = sublist[0]
    #     t_end = sublist[1]
    #     T_step = sublist[2]
    #     if (t_end-t_start) % T_step != 0:
    #         ramp_check = False
    #         print('\nTemperature ramp is not set up correctly!')
    #         print(f'--> List n° {count} contains an invalid temperature step ({T_step})\n')
    #     #check if each list has the right number of elements
    #     num_el = len(sublist)
    #     if num_el != 4:
    #         ramp_check = False
    #         print('\nTemperature ramp is not set up correctly!')
    #         print(f'--> List n° {count} contains only {num_el} elements!\n')
    # #if one condition is not satisfied, exit the program
    # if not ramp_check:
    #     print(f'\nYour ramp: {temp_set}\nThe right way: [[T_start (K), T_end (K), T_step (K), step_len (ns)],]\n')
    #     sys.exit(0)

    # global T_start
    # T_start = temp_set[0][0]
    # global T_stop
    # T_stop = temp_set[-1][1]

    # global timestep
    # timestep = args.timestep
    # global dcdfreq
    # dcdfreq = args.dcdfreq
    # global min_steps
    # min_steps  = args.min_steps
    # global equil1_len
    # equil1_len = args.equil1_len
    # global equil2_len
    # equil2_len = args.equil2_len
    # global resume
    # if args.resume == 'yes':
    #     resume = True
    # elif args.resume == 'no':
    #     resume = False
    # else:
    #     sys.exit('invalid resume settings')
    # global stride
    # stride = args.stride

    # global conversion_factor
    # conversion_factor = timestep * dcdfreq * stride / 1000000

    # global dryer
    # if args.dryer:
    #     dryer = 'yes'
    # else:
    #     dryer = ''
    # global smooth
    # smooth = args.smooth
    # global device
    # device = args.device
    # global n_procs
    # n_procs = args.n_procs
    # global vmd
    # #check if provided vmd path is correct: if not, search for local installation of vmd and use that instead
    # vmd_check = True
    # #control first if vmd path is provided
    # try:
    #     vmd = os.path.abspath(args.vmd_path)
    # except Exception:
    #     print('\nVMD path missing! (check your config file)\n')
    #     vmd_check = False
    # #control if provided path is a valid path
    # if vmd_check and not os.path.isfile(vmd):
    #     print(f'\n{vmd} is not a valid path\n')
    #     vmd_check = False
    # #control if provided vmd path refers to a vmd installation
    # if vmd_check and os.path.isfile(vmd):
    #     exe = vmd.split('/')[-1]
    #     if 'vmd' not in exe:
    #         print(f'\n{vmd} is not a valid VMD executable\n')
    #         vmd_check = False
    # if not vmd_check:
    #     #if vmd is not installed on local machine, exit from the program
    #     import subprocess
    #     try:
    #         vmd_installed_path = str(subprocess.check_output(['which','vmd']))[2:-3]
    #     except Exception:
    #         print('\nVMD is not installed on your machine!\n')
    #         sys.exit(0)
    #     print(f'\nFound existing installation of VMD at {vmd_installed_path}')
    #     print(f'Using {vmd_installed_path}\n')
    #     vmd = vmd_installed_path

    # # write config file with user-defined parameters for reproducibility reason
    # vars_file = f'''
    # [Defaults]

    # #system preparation
    # protein_name = {protein_name}
    # ligand_name = {ligand_name}
    # ligand_charge = {ligand_charge}
    # padding = {padding}
    # iso = {args.iso}

    # #simulation setup
    # temp_ramp = {temp_set}
    # timestep = {timestep}
    # dcdfreq = {dcdfreq}
    # min_steps  = {min_steps}
    # equil1_len = {equil1_len}
    # equil2_len = {equil2_len}
    # resume = {resume}

    # #postprocessing & analysis
    # stride = {stride}
    # dryer = {args.dryer}
    # smooth = {smooth}

    # #hardware settings
    # device = {device}
    # n_procs = {n_procs}

    # #external dependencies
    # vmd = {vmd}
    # '''

    # with open('vars.dat','w') as f:
    #     f.write(vars_file)

    # # print settings used for the current ttmd run, as stored in the 'vars.dat' file
    # print('\n** Parameters for your simulations were stored in vars.dat **\n')
    # print('\n#######################################################\n')
    # print(vars_file)
    # print('\n#######################################################\n')



'''
████████████████████████████████████████████████████████████████████████████████████████████████████
'''

### MULTIPROCESSING FUNCTION AND MODULES

class parallelizer(object):
    ### base class for multiprocessing
    def __init__(self, args, func, n_procs, desc):
        ### function initialization
        self.n_procs = n_procs
        self.args = args
        self.func = func
        self.desc = desc

    def start(self):
        pass

    def end(self):
        pass

    def run(args, func, num_procs, desc):
        return MultiThreading(args, func, n_procs, desc)
        ### run takes 4 arguments:
            # list of tup(args) for each spawned process
            # name of the function to be multiprocessed
            # number of process to spwan
            # description for the progression bar



def MultiThreading(args, func, n_procs, desc):
    results = []
    tasks = []
    for index,item in enumerate(args):
        task = (index, (func, item))
        ### every queue objects become indexable
        tasks.append(task)
    ### step needed to rethrieve correct results order
    results = start_processes(tasks, n_procs, desc)
    return results



def start_processes(inputs, n_procs, desc):
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
        args = [(task_queue, done_queue)]
        multiprocessing.Process(target=worker, args=args[0]).start()
        ### spawn (n_proc) worker function, that takes queue objects as args

    results = []
    for i in range(len(inputs)):
        results.append(done_queue.get())
        pbar.update(1)
        ### done_queue and progress bar update for each done object

    for i in range(n_procs):
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



'''
████████████████████████████████████████████████████████████████████████████████████████████████████
'''

class _time:
    def __init__(self):
        u = mda.Universe(top, trj)
        self.len_trj = len(u.trajectory)
        self.time = []
        self.frames = []
        for i in range(self.len_trj):
            self.frames.append(i)

            ns = round(i * cf, 1)
            self.time.append(ns)

time = _time()



class _select:

    def __init__(self):

        if 'protein' in receptorSel:
            self.sr = f'({receptorSel}) and backbone'
            self.cdmr_sel = f'{receptorSel} and resid {receptorResids}'

        elif 'nucleic' in receptorSel:
            self.sr = f'({receptorSel} and nucleicbackbone'
            self.cdmr_sel = f'{receptorSel} and resid{receptorResids}'

        else:
            self.sr = receptorSel
            self.cdmr_sel = receptorSel
        
    
        if 'protein' in ligandSel:
            self.sl = f'({ligandSel}) and backbone'
            self.cdml_sel = f'{ligandSel} and resid {ligandResids}'
            self.small_mol = False

        elif 'nucleic' in ligandSel:
            self.sl = f'({ligandSel}) and nucleicbackbone'
            self.cdml_sel = f'{ligandSel} and resid {ligandResids}'
            self.small_mol = False

        else:
            self.sl = ligandSel
            self.cdml_sel = ligandSel
            self.small_mol = True


        self.u = mda.Universe(top, trj)


        if ref_pdb != '':
            self.ref = mda.Universe(ref_pdb)
        else:
            self.ref = self.u

select = _select()



class _resids():

    def __init__(self):
        self.calculate = True
        self.dict = self.residue_dict()



    def get_top_contacts(self):
        #### this function parses the MD trajectory and returns the number of contacts between the ligand and each receptor residue and the correct label with resname and correct numeration

        sel_list = [[receptorSel, ligandSel, numResidRec, 'receptor', numShiftRec]]

        if select.small_mol == False:
            sel_list.append([ligandSel, receptorSel, numResidLig, 'ligand', numShiftLig])


        for l in sel_list:

            if not os.path.exists(f'{_wd}/contacts_{l[3]}'):

                print('\nCalculating receptor-ligand contacts...')

                contactSel = f"({l[0]}) and same residue as around {distanceCutoff} ({l[1]})"

                contactsList = []

                ### iterate through each trajectory frame
                for ts in select.u.trajectory:
                    #### create a ResidueGroup containing residues that are in contact with the ligand
                    contacts = select.u.select_atoms(contactSel).residues
                    contactsResidsList = []

                    for r in contacts:
                        contactsResidsList.append(r.resid)

                    contactsList.extend(contactsResidsList)


                #### create a sorted list of all residues in contact with the associated number of contacts
                count_list = sorted(Counter(contactsList).items(), key = lambda x: x[1], reverse=True)

                count = l[2]
                if len(count_list) < l[2]:
                    count = len(count_list)
                
                #### extract numResid best contacts and sort list
                top_list = [x[0] for x in count_list[:count]]
                resnum_list = sorted(top_list, key=lambda x: int(x))

                resnames = []

                #### extract resname from top contacts resids
                for r in resnum_list:
                    res = select.u.select_atoms(f'{l[0]} and resid {r}').residues
                    resname = str([x.resname for x in res]).lstrip("\'\[").rstrip("\'\]")
                    resnames.append(resname)

                #### write contacts file
                with open(f'{_wd}/contacts_{l[3]}', 'w') as f:
                    f.write('resnum,label,name_sel\n')
                    for resnum, resname in zip(resnum_list, resnames):
                        f.write(f'{resnum},{resname} {resnum + l[4]},{l[3]}\n')


        self.calculate = False



    def residue_dict(self):

        dict = {}

        if self.calculate == True:
            self.get_top_contacts()

        sel_list = ['receptor']

        if select.small_mol == False:
            sel_list.append('ligand')

        for f in sel_list:
            d = pd.read_csv(f'{_wd}/contacts_{f}')

            r = d['resnum']
            l = d['label']
            n = d['name_sel']

            for x,y,z in zip(r,l,n):

                dict[x] = [y, z]

        return dict

resids = _resids()



class _style:
    def __init__(self):
        if transparent == True:
            self.facecolor = 'transparent'

        elif transparent == False:
            if theme == 'light':
                self.facecolor = 'white'
            elif theme == 'dark':
                self.facecolor = 'transparent'

        if theme == 'light':
            self.ax_c = 'black'
            self.font_c = 'black'

        elif theme == 'dark':
            self.ax_c = 'white'
            self.font_c = 'white'



style = _style()


'''
████████████████████████████████████████████████████████████████████████████████████████████████████
'''

def trajectory_blocks():
    u = mda.Universe(top,trj)
    if not os.path.exists(f'{_wd}/splitted_trj'):
        os.mkdir(f'{_wd}/splitted_trj')

    n_frames = u.trajectory.n_frames
    frames_per_block = n_frames // n_procs
    blocks = [[(i * frames_per_block,), ((i + 1) * frames_per_block,), (top,), (trj,),]for i in range(n_procs - 1)]
    blocks.append([((n_procs - 1) * frames_per_block,), (n_frames,), (top,), (trj,),])

    block_names = []
    for b in blocks:
        first = list(b[0])[0]
        last = list(b[1])[0]
        block_names.append(f'{_wd}/splitted_trj/block_{first}_{last}.dcd')

    return [blocks, block_names]



def write_blocks(*args):
    first = list(args[0])[0]
    last = list(args[1])[0]
    topology = list(args[2])[0]
    trajectory = list(args[3])[0]
    
    block_name = f'block_{first}_{last}.dcd'

    if not os.path.exists(f'{_wd}/splitted_trj/{block_name}'):
        u = mda.Universe(topology, trajectory)
        
        with mda.Writer(f'{_wd}/splitted_trj/{block_name}', u.atoms.n_atoms) as W:
            for ts in u.trajectory[first:last]:
                W.write(u.atoms)

    return (block_name,)



def split_trj():
    trj_blocks = trajectory_blocks()

    already_blocks = glob.glob('block_*', root_dir=f'{_wd}/splitted_trj')

    for n in already_blocks:
        if not n in trj_blocks[1]:
            os.system(f'rm {_wd}/splitted_trj/{n}')

    to_do = []
    splitted = []

    for tup,n in zip(trj_blocks[0], trj_blocks[1]):
        if not n in already_blocks:
            to_do.append(tup)
        else:
            splitted.append((n,))

    if to_do != []:
        s = parallelizer.run(to_do, write_blocks, n_procs, 'Splitting trajectory')
        for x in s:
            splitted.append(x)

    return splitted



def colorbar_quantile(colorbar, color_list):
    cm = plt.cm.get_cmap(colorbar)
    sorted_list = np.sort(color_list)
    #### use numpy to calculate first quartile and third quartile
    vmin = np.nanquantile(sorted_list, 0.02)
    vmax = np.nanquantile(sorted_list, 0.98)

    return cm, vmin, vmax



def mount_panel(a,b,c,d, filename):
    os.system(f'montage -tile 2x2 -geometry 1920x1080 {a} {b} {c} {d} {filename}.png')
    return f'{filename}.png'



def rmsd_or_cdm():
    rmsd = f'{_wd}/geometry/rmsd_ref'
    cdm = f'{_wd}/geometry/cdm'

    df_dist = ''
    col = ''
    label = ''
    box_label = ''

    null = False

    if ref_bool == True:
        if os.path.exists(rmsd):
            df_dist = pd.read_csv(rmsd, sep = ',')
            col = 'rmsd'
            label = 'Ligand RMSD$_{backbone}$ to reference ($\AA$)'
            box_label ='RMSD'


    elif ref_bool == False:
        if os.path.exists(cdm):
            df_dist = pd.read_csv(cdm, sep=',')
            col = 'cdmdistance'
            label = 'dcm$_{bs-lig}$ ($\AA$)'
            box_label ='CDM Distance'

    else:
        null = True


    return df_dist, col, label, box_label, null



'''
████████████████████████████████████████████████████████████████████████████████████████████████████
'''


def rmsd(alignto, selection, groupsel, filename):
    R = mda.analysis.rms.RMSD(select.u, alignto,
                            select=selection,
                            groupselections=[groupsel])
    R.run()
    rmsd = R.results.rmsd.T[3]    # transpose makes it easier for plotting

    with open(filename, 'w') as f:
        f.write('frame,ns,rmsd\n')
        for frame, ns, value in zip(time.frames, time.time, list(rmsd)):
            f.write(f'{frame},{ns},{value}\n')

    df_rmsd = pd.DataFrame({'frame':time.frames, 'ns':time.time, 'rmsd':rmsd})
    
    return df_rmsd



def define_rmsd(filename, alignto, selection, groupsel, label):
    if not os.path.exists(filename):
        df_rmsd = rmsd(alignto, selection, groupsel, filename)
    else:
        df_rmsd = pd.read_csv(filename, sep=',')

    plotRMSD(df_rmsd, filename, label, 'all')
    
    if video == True:
        if not os.path.exists(f'{filename}_frames'):
            os.mkdir(f'{filename}_frames')

        os.chdir(f'{filename}_frames')

        mp_list = []
        for t in time.frames:
            if not os.path.exists(f'{filename}_{t}.png'):
                mp_list.append([df_rmsd, filename, label, t])

        parallelizer.run(mp_list, plotRMSD, n_procs, f'Writing {filename} frames')

        os.chdir('..')            



def run_rmsd():
    print('\nCalculating receptor RMSD...')
    if not os.path.exists('geometry'):
        os.mkdir('geometry')

    os.chdir('geometry')
    if ref_pdb != '':

        filename = 'rmsd_ref'

        if select.small_mol == True:
            label = 'Ligand RMSD to reference'
        elif select.small_mol == False:
            label = 'Ligand RMSD$_{backbone}$ to reference'

        define_rmsd(filename, select.ref, select.sr, select.sl, label)

    filerec = 'rmsd_rec'
    labelrec = 'Receptor RMSD$_{backbone}$'
    define_rmsd(filerec, select.u, select.sr, select.sr, labelrec)

    if select.small_mol == False:
        filelig = 'rmsd_lig'
        labellig = 'Ligand RMSD$_{backbone}$'
        define_rmsd(filelig, select.u, select.sl, select.sl, labellig)

    os.chdir(_wd)



def plotRMSD(df, filename, label, index):
    if not os.path.exists(f'{filename}.png'):
        df_x = list(df['ns'])
        df_y = list(df['rmsd'])

        if index == 'all' or index == -1:
            x = df_x
            y = df_y

        else:
            i = index + 1
            x = df_x[:i]
            y = df_y[:i]

        #### this function calculate and plots the receptor RMSD
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(x, y, 'k-', linewidth=0.75, color='purple')
        # ax.legend(loc="best")
        ax.set_xlabel("Time (ns)", fontsize=fontsize)
        ax.set_ylabel(r"RMSD ($\AA$)", fontsize=fontsize)
        ax.set_title(label, fontsize=fontsize + 2)
        ax.set_xlim(0, df_x[-1])
        ax.set_ylim(0, max(df_y)+max(df_y)*5/100)
        plt.tight_layout()
        fig.savefig(f'{filename}_{index}.png', dpi=300)

    return filename




##################################################

def run_cmdist():
    print('\nCalculating center of mass distance...')

    if not os.path.exists('geometry'):
        os.mkdir('geometry')

    os.chdir('geometry')

    filename = 'cmd'

    cmd_list = []

    if not os.path.exists(filename):
        for ts in select.u.trajectory:
            receptorCMD = select.u.select_atoms(select.cdmr_sel).center_of_mass()
            ligandCMD = select.u.select_atoms(select.cdml_sel).center_of_mass()

            distance = np.linalg.norm(ligandCMD - receptorCMD)
            cmd_list.append(distance)

        df_cdm = pd.DataFrame({'ns':time.time, 'cmdistance':cmd_list})

        with open(filename, 'w') as f:
            f.write('frame,ns,cmdistance\n')
            for frame, ns, cmd in zip(time.frames, time.time, cmd_list):
                f.write(f'{frame},{ns},{cmd}\n')

    else:
        df_cmd = pd.read_csv(filename, sep=',')

    plotCMD(df_cmd, 'all')

    if video == True:
        if not os.path.exists(f'{filename}_frames'):
            os.mkdir(f'{filename}_frames')

        os.chdir(f'{filename}_frames')
        mp_list = []
        for t in time.frames:
            mp_list.append([df_cmd, t])

        parallelizer.run(mp_list, plotCMD, n_procs, f'Writing {filename} frames')

    os.chdir(_wd)



def plotCMD(cmd_list, index):
    df_x = cmd_list['ns']
    df_y = cmd_list['cmdistance']

    y_max = 0
    for y in df_y:
        if y > y_max:
            y_max = y
    
    if index == 'all' or index == -1:
        x = df_x
        y = df_y

    else:
        i = index + 1
        x = df_x[:i]
        y = df_y[:i]

    #### this function calculate and plots the distance between the center of mass of the ligand and the binding site at each frame

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y, 'k-')
    ax.set_xlabel("Time (ns)", fontsize=fontsize)
    ax.set_ylabel(r"cdm$_{rec-lig}$ distance ($\AA$)", fontsize=fontsize)
    ax.set_xlim(0, df_x[len(df_x) - 1])
    ax.set_ylim(0, y_max+y_max*5/100)
    ax.set_title('CMD$_{rec-lig}$ distance', fontsize=fontsize + 2)
    plt.tight_layout()
    fig.savefig(f'cdm_distance_{index}.png', dpi=300)



##################################################
    
def rmsf(*args):
    list_args = list(args)
    u = list_args[0]
    n = list_args[1]
    numShift = list_args[2]
    sel_string = list_args[3]

    selection = u.atoms.select_atoms(sel_string)
    residues = np.unique(selection.resnums)

    rmsf_x_residue = []

    for res in residues:
        if n == 0:
            pass
        else:
            sel = selection.atoms.select_atoms(f'resid {res}')
            R = mda.analysis.rms.RMSF(sel).run(start=0, stop=n)
            r = R.results.rmsf
            list_rmsf = list(r)
            mean_rmsf = sum(list_rmsf)/len(list_rmsf)
            rmsf_x_residue.append([n, res + numShift, mean_rmsf])

    return rmsf_x_residue


def mp_rmsf(sel_string, numShift, filename):
    if not os.path.exists(f'rmsf_x_residue_{filename}'):
        print(f'\nTrajectory alignment selection = \'{sel_string}\'')

        align.AlignTraj(select.u, select.u, select=sel_string, in_memory = True).run()

        rmsf_input = []
        
        for ts in select.u.trajectory:
            n = ts.frame
            rmsf_input.append((select.u, n, numShift, sel_string))

        rmsf_x_residue = parallelizer.run(rmsf_input, rmsf, n_procs, f'Frames')
        rmsf_x_residue.pop(0)

        with open(f'rmsf_x_residue_{filename}', 'w') as f:
            f.write('frame,ns,residue,rmsf\n')
            for ts in rmsf_x_residue:
                for ts, res, value in ts:
                    f.write(str(f'{ts},{round(ts * cf, 1)},{res},{value}\n'))

    df_rmsf = pd.read_csv(f'rmsf_x_residue_{filename}', sep=',')

    return df_rmsf



def run_rmsf():
    if not os.path.exists('geometry'):
        os.mkdir('geometry')

    os.chdir('geometry')

    print('\nCalculating per residue RMSF per frame...')
    rmsf_r = mp_rmsf(select.sr, numShiftRec, 'rec')
    plot_tot_rmsf(rmsf_r, 'Receptor RMSF$_{backbone}$', 'rmsf_rec')
    plot_rmsf_timeline(rmsf_r, 'Receptor RMSF$_{backbone}$', 'rmsf_rec')

    rmsf_l = mp_rmsf(select.sl, numShiftLig, 'lig')
    plot_tot_rmsf(rmsf_l, 'Ligand RMSF$_{backbone}$', 'rmsf_lig')
    plot_rmsf_timeline(rmsf_l, 'Ligand RMSF$_{backbone}$', 'rmsf_lig')

    os.chdir(_wd)



def plot_tot_rmsf(df_rmsf, label, filename):
    residue = df_rmsf['residue']
    x = []
    y = []

    for res in residue.unique():
        last = 0
        for r, rmsf in zip(df_rmsf['residue'], df_rmsf['rmsf']):
            if r == res:
                last = rmsf

        x.append(last)
        y.append(res) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'k-', linewidth=0.75, color='purple')
    # ax.legend(loc="best")
    ax.set_xlabel('RMSF (Å)', fontsize=14)
    ax.set_ylabel('Residue number', fontsize=14)
    ax.set_xlim(0, max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_title(label, fontsize=16)
    plt.tight_layout()
    fig.savefig(f'{filename}.png', dpi=300)


def plot_rmsf_timeline(df_rmsf, label, filename):

    ### RMSF through time per residue

    plt.figure()

    x = df_rmsf['ns'].to_numpy()
    y = df_rmsf['residue'].to_numpy()
    rmsf = df_rmsf['rmsf'].to_numpy()
    cm, vmin, vmax = colorbar_quantile('Greens', rmsf)
    plt.scatter(x, y, c=rmsf, s=15, cmap=cm, marker='s', vmin=vmin, vmax=vmax, linewidths= 0)
    cbar = plt.colorbar()
    cbar.set_label('RMSF (Å)', rotation=270, labelpad=15)
    plt.title(label, fontsize=16)
    plt.ylabel('Residue number', fontsize=14)
    plt.xlabel('Time (ns)', fontsize=14)
    # plt.xlim(min(x), max(x))
    # plt.ylim(min(y), max(y))
    plt.tight_layout()
    plt.savefig(f'{filename}_timeline.png', dpi=300)
    plt.close()



##################################################

def rgyr(sel_string, filename):
    if not os.path.exists(f'rgyr_{filename}'):

        sel = select.u.select_atoms(sel_string)
        gyr = []

        for ts in select.u.trajectory:
            rg = sel.radius_of_gyration()
            gyr.append(rg)

        with open(f'rgyr_{filename}', 'w') as f:
            f.write('frame,ns,rgyr\n')
            for t,n,g in zip(time.frames, time.time, gyr):
                f.write(f'{str(t)},{str(n)},{str(g)}\n')

        df_rgyr = pd.DataFrame({'frame':time.frames, 'ns':time.time, 'rgyr':gyr})

    else:
        df_rgyr = pd.read_csv(f'rgyr_{filename}', sep=',')

    return df_rgyr



def run_rgyr():
    if not os.path.exists('geometry'):
        os.mkdir('geometry')

    os.chdir('geometry')

    print('\nCalculating radius of gyration...')
    rgyr_r = rgyr(receptorSel, 'rec')
    plot_rgyr(rgyr_r, "Receptor Radius of Gyration", 'rgyr_rec')

    rgyr_l = rgyr(ligandSel, 'lig')
    plot_rgyr(rgyr_l, "Ligand Radius of Gyration", 'rgyr_lig')

    os.chdir(_wd)



def plot_rgyr(df_rgyr, label, filename):
    plt.figure()
    ax = plt.subplot(111)

    x = list(df_rgyr['ns'])
    y = list(df_rgyr['rgyr'])
    
    xnew = np.linspace(min(x), max(x), 300) 
    spl1 = make_interp_spline(x, y, k=3)
    power_smooth1 = spl1(xnew)
    ax.plot(xnew, power_smooth1, linewidth=0.75, color='purple')
    ax.set_xlabel("Time (ns)", fontsize=14)
    ax.set_ylabel(r"radius of gyration $R_G$ ($\AA$)", fontsize=14)
    ax.set_xlim(0, max(x))
    plt.title(label, fontsize=16)
    plt.tight_layout()
    ax.figure.savefig(f'{filename}.png' ,dpi=300)


##################################################

def run_ERMSD():
    #### this function exploits the barnaba python package to calculate ERMSD
    print('\nCalculating ERMSD using barnaba...')

    os.chdir('geometry')

    if ref_pdb != '':
        native = ref_pdb
    else:
        native = top

    if not os.path.exists('bb_ERMSD') or not os.path.exists('bb_RMSD'):
    # calculate eRMSD between native and all frames in trajectory
        ermsd = bb.ermsd(native,trj,topology=top)
        rmsd = bb.rmsd(native,trj,topology=top)

        with open('bb_ERMSD', 'w') as f:
            f.write('frame,ns,ermsd\n')
            for t,ns,e in zip(time.frames, time.time, ermsd):
                f.write(f'{t},{ns},{e}\n')

        with open('bb_RMSD', 'w') as f:
            f.write('frame,ns,rmsd\n')
            for t,ns,e in zip(time.frames, time.time, rmsd):
                f.write(f'{t},{ns},{e}\n')

        df_ermsd = pd.DataFrame({'frame':time.frames, 'ns':time.time, 'ermsd':ermsd})
        df_rmsd = pd.DataFrame({'frame':time.frames, 'ns':time.time, 'rmsd':rmsd})

    else:
        df_ermsd = pd.read_csv('bb_ERMSD', sep=',')
        df_rmsd = pd.read_csv('bb_RMSD', sep=',')

    plotERMSD(df_ermsd, df_rmsd)

    os.chdir(_wd)



def plotERMSD(df_ermsd, df_rmsd):
    # plot time series

    a = 'ERMSD_vs_time.png'
    plt.figure()
    plt.plot(df_ermsd['ns'], df_ermsd['ermsd'], linewidth=0.75, color='orange')
    plt.ylabel("eRMSD from native")
    plt.xlabel('Time (ns)')
    plt.tight_layout()
    plt.savefig(a, dpi=300)
    plt.clf()

    # make histogram

    b = 'ERMSD_distribution.png'
    plt.figure()
    plt.hist(df_ermsd['ermsd'],density=True,bins=50, stacked=True, alpha=0.5, color='orange')
    plt.xlabel("eRMSD from native")
    plt.ylabel('Probability density')
    plt.tight_layout()
    plt.savefig(b, dpi=300)
    plt.clf()
    
    # plot time series

    c = 'RMSD_vs_time.png'
    plt.figure()
    plt.plot(df_rmsd['ns'], df_rmsd['rmsd'], linewidth=0.75)
    plt.ylabel("RMSD from native (nm)")
    plt.xlabel('Time (ns)')
    plt.tight_layout()
    plt.savefig(c,dpi=300)
    plt.clf()

    # make histogram
    d = 'RMSD_distribution.png'
    plt.hist(df_rmsd['rmsd'],density=True,bins=50, stacked=True, alpha=0.5)
    plt.xlabel("RMSD from native (nm)")
    plt.ylabel('Probability density')
    plt.tight_layout()
    plt.savefig(d, dpi=300)
    plt.clf()
    
    # combined plot
    plt.xlabel("eRMSD from native")
    plt.ylabel("RMSD from native (nm)")
    plt.axhline(0.4,ls = "--", c= 'k')
    plt.axvline(0.7,ls = "--", c= 'k')
    plt.scatter(df_ermsd['ermsd'], df_rmsd['rmsd'], s=2.5)
    plt.tight_layout()
    plt.savefig('barnaba.png',dpi=300)
    plt.clf()
    
    # merged panel
    mount_panel(a,b,c,d, 'merged_barnaba')



def final_mount():
    #### this function mounts a 4 tile panel for receptor geometric analysis
    os.chdir('geometry')
    for mol in ['rec', 'lig']:
        a = f'rmsd_{mol}_all.png'
        b = f'rgyr_{mol}.png'
        c = f'rmsf_{mol}.png'
        d = f'rmsf_{mol}_timeline.png'

        mount_panel(a, b, c, d, f'merged_{mol}')

    os.chdir(_wd)



'''
████████████████████████████████████████████████████████████████████████████████████████████████████
'''

def mmgbsa(traj):
    name = traj.rstrip('.dcd').lstrip('splitted_trj/block_')
    trj_name = f'{_wd}/splitted_trj/{traj}'
    files = [f'gb_{name}.dat', f'gb_frame_{name}.dat']


    if not os.path.exists(f'gb_{name}.dat') or not os.path.exists(f'gb_frame_{name}.dat'):
        if not os.path.exists(name):
            os.mkdir(name)
    
        os.chdir(name)

        os.system(f'$AMBERHOME/bin/MMPBSA.py -i {_wd}/MMGBSA/mmgbsa.in -o gb_{name}.dat -eo gb_frame_{name}.dat -cp {itop} -rp {rec_parm} -lp {lig_parm} -y {trj_name} > /dev/null 2>&1')

        os.chdir('..')

        for f in files:
            os.system(f'cp {name}/{f} .')
        
        os.system(f'rm -r {name}')

    return files



def run_MMGBSA():
    #### this function calculates MMGBSA interaction energy between receptor and ligand
    print('\nCalculating MMGBSA interaction energy...')

    if not os.path.exists('MMGBSA'):
        os.mkdir('MMGBSA')

    os.chdir('MMGBSA')

    if not os.path.exists('gb_DELTA'):
        splitted = split_trj()

        mmgbsa_in = '''
    Input file for running PB and GB
    &general
    verbose=1,
    /
    &gb
    igb=5, saltcon=0.100
    /
    '''
        with open('mmgbsa.in','w') as f:
            f.write(mmgbsa_in)
        
        files = parallelizer.run(splitted, mmgbsa, n_procs, 'Running MMGBSA')

        lines_to_write = []

        for file in files:
            f = file[1]
            with open(f, 'r') as r:
                lines = r.readlines()
                lines = lines[:-2]
                for i,line in enumerate(lines):
                    if line == 'DELTA Energy Terms\n':
                        start = i + 2
                        break

                for line in lines[start:]:
                    lines_to_write.append(line)

        delta_lines = []
        best_delta = 0
        best_frame = ''

        for i,l in enumerate(lines_to_write):
            delta = float(l.split(',')[7].rstrip('\n'))
            if float(delta) < best_delta:
                best_delta = float(delta)
                best_frame = i
            delta_lines.append(delta)

        with open('gb_DELTA', 'w') as f:
            f.write('frame,ns,delta\n')
            for i,l in enumerate(delta_lines):
                f.write(f'{time.frames[i]},{time.time[i]},{l}\n')

        with open('delta_min', 'w') as f:
            f.write(f'frame,ns,delta\n{best_frame},{best_frame*cf},{best_delta}')

        df_delta = pd.DataFrame({'frames':time.frames, 'ns':time.time, 'delta':delta_lines})

    else:
        df_delta = pd.read_csv('gb_DELTA', sep=',')

    mmgbsa_time(df_delta)
    mmgbsa_dist(df_delta)
    plot_mmgbsa_timexdist(df_delta)
    
    os.chdir(_wd)


        
def plot_mmgbsa_timexdist(df_delta):
    print('Plotting mmgbsa vs distance vs time')

    df_dist, col, label, box_label, null = rmsd_or_cdm()

    if null == True:
        print('No distance files found: execute geometric calculations first!')
        pass

    else:
        x = time.time
        x_f = time.frames
        y = df_dist[col]
        z = df_delta['delta']

        df_min = pd.read_csv('delta_min')
        frame_min = list(df_min['frame'])[0]
        ns_min = list(df_min['ns'])[0]
        gb_min = list(df_min['delta'])[0]
        dist_min = list(df_dist[col])[frame_min]
        

        color_list = np.array(z)

        cm, vmin, vmax = colorbar_quantile('RdYlBu_r', color_list)

        plt.figure()
        
        plt.scatter(x, y, c=z, cmap=cm, vmin=vmin, vmax=vmax)
        plt.scatter(ns_min, dist_min, c=gb_min, s=70, cmap=cm, vmin=vmin, vmax=vmax, edgecolors='limegreen', linewidths=2)
        t = plt.text(max(x), max(y), f'Time = {ns_min} (ns)\n{box_label} = {round(dist_min,2)} ($\AA$)\nMMGBSA = {round(gb_min,2)} (kcal/mol)', fontsize=10, in_layout=True, ma='left', va='top', ha='right', bbox=dict(boxstyle="round,pad=0.5", fc="None", ec="limegreen", lw=1))
        cbar = plt.colorbar()
        cbar.set_label('MMGBSA (kcal/mol)', rotation=270, labelpad=15)
        plt.title('Interaction Energy Landscape', fontsize=16)
        plt.ylabel(label, fontsize=14)
        plt.xlabel('Time (ns)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'mmgbsa_vs_{col}.png',dpi=300)
        


def mmgbsa_dist(df_delta):
    print('Plotting mmgbsa vs distance')
    df_dist, col, label, box_label, null = rmsd_or_cdm()

    if null == True:
        print('No distance files found: execute geometric calculations first!')
        pass

    else:
        x = df_dist[col]
        y = df_delta['delta']
        color_list = np.array(y)

        cm, vmin, vmax = colorbar_quantile('RdYlBu_r', color_list)

        plt.figure()
        plt.scatter(x, y, c=y, cmap=cm, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar()
        cbar.set_label('MMGBSA (kcal/mol)', rotation=270, labelpad=15)
        plt.title('Interaction Energy Landscape', fontsize=16)
        plt.ylabel('MMGBSA (kcal/mol)', fontsize=14)
        plt.xlabel(label, fontsize=14)
        plt.tight_layout()
        plt.savefig('mmgbsa_vs_distance.png',dpi=300)



def mmgbsa_time(df_delta):
    print('Plotting mmgbsa vs time')
    x = time.time
    y = df_delta['delta']
    color_list = np.array(y)

    cm, vmin, vmax = colorbar_quantile('RdYlBu_r', color_list)

    plt.figure()
    plt.scatter(x, y, c=y, cmap=cm, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar()
    cbar.set_label('MMGBSA (kcal/mol)', rotation=270, labelpad=15)
    plt.title('Interaction Energy Profile', fontsize=16)
    plt.ylabel('MMGBSA (kcal/mol)', fontsize=14)
    plt.xlabel('Time (ns)', fontsize=14)
    plt.tight_layout()
    plt.savefig('mmgbsa_vs_time.png',dpi=300)



'''
████████████████████████████████████████████████████████████████████████████████████████████████████
'''

def run_ie_namd():
    #### this function calculates interaction energy between receptor and ligand with NAMD
    print('\nCalculating interaction energy (NAMD)...')

    output_basename = 'interactionEnergy'

    if not os.path.exists('InteractionEnergy'):
        os.mkdir('InteractionEnergy')

    os.chdir('InteractionEnergy')

    if not os.path.exists('interactionEnergy.dat'):
 
        vmdFile = 'interactionEnergy.tcl'

        with open(vmdFile, 'w') as f:
            f.write(f'''
mol new {itop}
mol addfile {trj} type dcd  first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all
set prot [atomselect top "{receptorSel}"]
set ligand [atomselect top "{ligandSel}"]
global env
set Arch [vmdinfo arch]
set vmdEnv $env(VMDDIR)
puts $vmdEnv
source $vmdEnv/plugins/noarch/tcl/namdenergy1.4/namdenergy.tcl
namdenergy -exe {namdPATH} -elec -vdw -sel $ligand $prot -ofile "{output_basename}.dat" -tempname "{output_basename}_temp" -ts {timestep * stride} -timemult {timestep} -stride {stride} -switch  7.5 -cutoff 9 -par {par}
quit''')

        os.system(f'vmd -dispdev text -e {vmdFile} > /dev/null 2>&1')

        os.remove(vmdFile)

    df_ie = pd.read_table(f'{output_basename}.dat', sep='\s+')

    ie_time(df_ie)
    ie_dist(df_ie)

    df_dist, col, label, box_label, null = rmsd_or_cdm()
    
    if null == True:
        print('No distance files found: execute geometric calculations first!')
        pass

    else:
        ie_dist_time(df_ie, df_dist, col, label, box_label, '')
        video_ie(df_ie)
    

    os.chdir(_wd)

      

def ie_time(df_ie):
    print('Plotting interaction energy vs time')
    x = time.time
    y = df_ie['Total']
    color_list = np.array(y)

    cm, vmin, vmax  = colorbar_quantile('RdYlBu_r', color_list)

    plt.figure()
    plt.scatter(x, y, c=y, cmap=cm, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar()
    cbar.set_label('Interaction Energy (kcal/mol)', rotation=270, labelpad=15)
    plt.title('Interaction Energy Profile', fontsize=16)
    plt.ylabel('Interaction Energy (kcal/mol)', fontsize=14)
    plt.xlabel('Time (ns)', fontsize=14)
    plt.tight_layout()
    plt.savefig('ie_vs_time.png',dpi=300)


    
def ie_dist(df_ie):
    print('Plotting interaction energy vs distance')
    df_dist, col, label, box_label, null = rmsd_or_cdm()

    if null == True:
        print('No distance files found: execute geometric calculations first!')
        pass

    else:
        x = df_dist[col]
        y = df_ie['Total']
        color_list = np.array(y)

        cm, vmin, vmax  = colorbar_quantile('RdYlBu_r', color_list)

        plt.figure()
        plt.scatter(x, y, c=y, cmap=cm, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar()
        cbar.set_label('Interaction Energy (kcal/mol)', rotation=270, labelpad=15)
        plt.title('Interaction Energy Landscape', fontsize=16)
        plt.ylabel('Interaction Energy (kcal/mol)', fontsize=14)
        plt.xlabel(label, fontsize=14)
        plt.tight_layout()
        plt.savefig('ie_vs_distance.png',dpi=300)



def video_ie(df_ie):
    print('\nPlotting interaction energy vs distance...')
    df_dist, col, label, box_label, null = rmsd_or_cdm()

    if video == True:
        if not os.path.exists(f'ie_frames'):
            os.mkdir(f'ie_frames')

        os.chdir(f'ie_frames')

        mp_list = []
        for t in time.frames:
            if not os.path.exists(f'ie_{t}.png'):
                mp_list.append([(df_ie), (df_dist), (col), (label), (box_label), (t)])
        parallelizer.run(mp_list, ie_dist_time, n_procs, f'Writing interaction energy frames')

        os.chdir(_wd + '/InteractionEnergy')

    else:
        pass



def ie_dist_time(df_ie, df_dist, col, label, box_label, index):
    
    df_x = time.time
    df_x_f = time.frames
    df_y = df_dist[col]
    z = df_ie['Total']

    color_list = np.array(z)
    cm, vmin, vmax  = colorbar_quantile('RdYlBu_r', color_list)

    y_max = 0
    for y in df_y:
        if y > y_max:
            y_max = y
    
    if index == '':
        print('\nPlotting interaction energy vs distance...')
        x = df_x
        y = df_y
        z = z

    else:
        i = index + 1
        x = df_x[:i]
        y = df_y[:i]
        z = list(z)[:i]

    f_min = 0
    ns_min = 0
    ie_min = 0

    for frame,t,ie in zip(df_ie['Frame'], df_ie['Time'], df_ie['Total']):
        if ie < ie_min:
            f_min = frame
            ns_min = t / 10**2
            ie_min = ie
    
    dist_min = list(df_y)[f_min]

    plt.figure()
    plt.scatter(x, y, c=z, cmap=cm, vmin=vmin, vmax=vmax)
    
    if index == '':
        plt.scatter(ns_min, dist_min, c=ie_min, s=70, cmap=cm, vmin=vmin, vmax=vmax, edgecolors='limegreen', linewidths=2)
        t = plt.text(max(x)-max(x)*5/100, max(y)-max(y)*5/100, f'Time = {ns_min} (ns)\n{box_label} = {round(dist_min,2)} ($\AA$)\nInteraction Energy = {round(ie_min,2)} (kcal/mol)', fontsize=10, in_layout=True, ma='left', va='top', ha='right', bbox=dict(boxstyle="round,pad=0.5", fc="None", ec="limegreen", lw=1))
    cbar = plt.colorbar()
    cbar.set_label('Interaction Energy (kcal/mol)', rotation=270, labelpad=15)
    plt.title('Interaction Energy Landscape', fontsize=16)
    plt.ylabel(label, fontsize=14)
    plt.xlabel('Time (ns)', fontsize=14)
    plt.xlim(0,df_x[-1])
    plt.ylim(0,max(list(df_y)))
    plt.tight_layout()
    plt.savefig(f'ie_{index}.png',dpi=300)

    return f'ie_{index}.png'


'''
████████████████████████████████████████████████████████████████████████████████████████████████████
'''

def calc_heatmap(sel1, sel2, name_sel):

    #### this function calculates elec, vdw and total energy between ligand and each protein residue defined in the previous steps
    print('\nCalculating per-residue interaction energy with NAMD...\n')

    if not os.path.exists('total.csv'):
        #### iter through each residue and calculate interaction energy with namdenergy plugin from vmd (namd2 exe required)
        mp_resids = []
        for i in resids.dict:
            if resids.dict[i][1] == name_sel:
                res = ([sel1, sel2, int(i)])
                mp_resids.append(res)

        parallelizer.run(mp_resids, calc_namd, n_procs, f'Calculating Interaction Energy {name_sel}')

        HeatmapProcessing(name_sel)



def calc_namd(sel1, sel2, resid):
    selection1 = f'{sel1} and resid {resid}'
    selection2 = f'{sel2}'
    output_basename = f'interactionEnergy_{resid}'
    vmdFile = f'interactionEnergy_{resid}.tcl'

    if not os.path.exists(f'{output_basename}.dat'):
        with open(vmdFile, 'w') as f:
            f.write(f'''mol new {itop}
    mol addfile {trj} type dcd  first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all
    set prot [atomselect top "{selection1}"]
    set ligand [atomselect top "{selection2}"]
    global env
    set Arch [vmdinfo arch]
    set vmdEnv $env(VMDDIR)
    puts $vmdEnv
    source $vmdEnv/plugins/noarch/tcl/namdenergy1.4/namdenergy.tcl
    namdenergy -exe {namdPATH} -elec -vdw -sel $ligand $prot -ofile "{output_basename}.dat" -tempname "{output_basename}_temp" -ts {dcdfreq} -timemult {timestep} -stride {stride} -switch  7.5 -cutoff 9 -par {par}
    quit''')

        os.system(f'vmd -dispdev text -e {vmdFile} > /dev/null 2>&1')

        os.remove(vmdFile)



def run_res_ie():

    sel_list = [[receptorSel, ligandSel, 'receptor']]

    if select.small_mol == False:
        sel_list.append([ligandSel, receptorSel, 'ligand'])

    for l in sel_list:

        if not os.path.exists(f'per_residue_{l[2]}'):
            os.mkdir(f'per_residue_{l[2]}')

        os.chdir(f'per_residue_{l[2]}')

        if not os.path.exists('total.csv'):
            calc_heatmap(l[0], l[1], l[2])
            
        define_heatmap(l[2])

        os.chdir(_wd)



def HeatmapProcessing(name_sel):
    #### this function combines the single namd output files into single csv that can be manipulated for plotting purpose
    print(f'\nProcessing {name_sel} output files...')

    #### create a list of all interaction energy files ordered by ascending resid number
    intFile = sorted(glob.glob('interactionEnergy_*.dat'), key=lambda x: int(os.path.basename(x).split('_')[-1].rstrip('.dat')))
    print(intFile)
    #### create a pandas dataframe to manipulate data from the interaction energy file
    resids = []
    total_dict = {}

    for f in intFile:
        r = os.path.basename(f).split('_')[-1].rstrip('.dat')
        resids.append(r)
        df_total = pd.read_table(f, sep='\s+')['Total']
        total_dict[r] = list(df_total)

    df = pd.DataFrame(total_dict)
    df.to_csv('total.csv', sep=',', header=True, index=False)
    


def define_heatmap(name_sel):
    print('\nPlotting Receptor Per-Residue Interaction Energy Heatmap...')

    df = pd.read_csv('total.csv', sep=',').T
    resid_correct = []

    for i in resids.dict:
        if resids.dict[i][1] == name_sel:
            resid_correct.append(resids.dict[i][0])

    df_plot = pd.DataFrame(df.values, columns=time.time, index=resid_correct)
    
    color_list = []
    for col in df.columns:
        for f in df[col]:
            color_list.append(f)

    cm, vmin, vmax  = colorbar_quantile('RdBu_r', color_list)

    plot_heatmap(df_plot, name_sel, cm, vmin, vmax, '')

    

    if video == True:
        if not os.path.exists(f'heatmap_frames'):
            os.mkdir(f'heatmap_frames')

        os.chdir(f'heatmap_frames')

        mp_list = []
        for t in time.frames:
            if not os.path.exists(f'heatmap_{t}.png'):
                mp_list.append([(df_plot), (name_sel), (cm), (vmin), (vmax), (t)])
        parallelizer.run(mp_list, plot_heatmap, n_procs, f'Writing interaction energy frames')

        os.chdir(_wd + '/InteractionEnergy')

    else:
        pass



def plot_heatmap(df_plot, name_sel, cm, vmin, vmax, index):
    #### this function creates the heatmap for each replica using the mask values previously calculated

    if index == '':
        columns_list = df_plot.columns

    else:
        columns_list = df_plot.columns[0:index+1]
    
    df = df_plot.isin(df_plot[columns_list])

    xtl = round(len(time.time)*20/100)

    #### create heatmap
    fig = plt.figure(facecolor='white')
    ax = sns.heatmap(df_plot, cmap=cm, center=0, mask=~df,vmin=vmin, vmax=vmax, xticklabels=xtl)
    ax.set(xlim=(0, time.frames[-1]))
    cbar_axes = ax.figure.axes[-1]
    cbar_axes.set_ylabel('Interaction Energy (Kcal/mol)', rotation=270, labelpad=15)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    ax.set_title(f'Per-Residue Interaction Energy ({name_sel})')
    ax.set_ylabel('Residue')
    ax.set_xlabel('Time (ns)')
    plt.tight_layout()
    fig.savefig(f'heatmap_{index}.png', dpi=300)

    return f'heatmap_{index}.png'



'''
████████████████████████████████████████████████████████████████████████████████████████████████████
'''



def run_matrix_namd():
    #### this function calculates a pairwise per-residue elec, vdw and total interaction energy value between receptor and ligand with NAMD
    print('\nCalculating pairwise per-residue interaction energy with NAMD...\n')

    if not os.path.exists('per_residue_matrix'):
        os.mkdir('per_residue_matrix')

    os.chdir('per_residue_matrix')

    mp_list = []

    for r in resids.dict:
        if resids.dict[r][1] == 'receptor':
            ligand_list = []
            for l in resids.dict:
                if resids.dict[l][1] == 'ligand':
                    ligand_list.append(l)
            mp_list.append([r, ligand_list])


    rr_filename_mean = parallelizer.run(mp_list, calc_matrix, n_procs, 'Calculating pairwise per-residue IE')

    plot_matrix(rr_filename_mean, '')

    if video == True:
        if not os.path.exists(f'matrix_frames'):
            os.mkdir(f'matrix_frames')

        os.chdir(f'matrix_frames')

        mp_list = []
        for t in time.frames:
            if not os.path.exists(f'_matrix_{t}.png'):
                mp_list.append([rr_filename_mean, t])

        parallelizer.run(mp_list, plot_matrix, n_procs, f'Writing matrix frames')

        os.chdir(_wd + '/per_residue_matrix')

    else:
        pass

    os.chdir(_wd)
    


def calc_matrix(rr, res_lig):
    filename = f'per_res_{rr}'

    if not os.path.exists(filename+'_'):
        basename_list = []

        for rl in res_lig:

            ligSel = f'({ligandSel}) and resid {rl}'

            output_basename = f'interactionEnergy_{rl}_{rr}'
            basename_list.append([rl, output_basename + '.dat'])

            protSel = f'({receptorSel}) and resid {rr}'
            vmdFile = f'interactionEnergy_{rl}_{rr}.tcl'

            if not os.path.exists(f'{output_basename}.dat'):
                with open(vmdFile, 'w') as f:
                    f.write(f'''mol new {itop}
        mol addfile {trj} type dcd  first 0 last -1 step 1 filebonds 1 autobonds 1 waitfor all
        set prot [atomselect top "{protSel}"]
        set ligand [atomselect top "{ligSel}"]
        global env
        set Arch [vmdinfo arch]
        set vmdEnv $env(VMDDIR)
        puts $vmdEnv
        source $vmdEnv/plugins/noarch/tcl/namdenergy1.4/namdenergy.tcl
        namdenergy -exe {namdPATH} -elec -vdw -sel $ligand $prot -ofile "{output_basename}.dat" -tempname "{output_basename}_temp" -ts {dcdfreq} -timemult {timestep} -stride {stride} -switch  7.5 -cutoff 9 -par {par}
        quit''')

                os.system(f'vmd -dispdev text -e {vmdFile} > /dev/null 2>&1')

                os.system(f'rm {vmdFile}')


        mean = []
        df = pd.DataFrame()

        for f in basename_list:
            total = pd.read_table(f[1], sep='\s+')['Total']
            m = total.mean()
            mean.append([f[0], m])

            df[f[0]] = total.T

        df.to_csv(filename, sep=',', index=False, index_label='ns')

    return [filename, rr, mean]



def process_values(rr_filename_mean, index):
    x = []
    y = []
    z = []
    color_list = []

    if index == '':
        for t in rr_filename_mean:
            rr = t[1]
            for arr in t[2]:
                rl = arr[0]
                v = arr[1]
                x.append(resids.dict[rr][0])
                y.append(resids.dict[rl][0])
                z.append(v)
                color_list.append(v)

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

                x.append(resids.dict[rr][0])
                y.append(resids.dict[int(col)][0])
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

    

def panel_mounting():

    if not os.path.exists(f'{_wd}/montage'):
        os.mkdir(f'{_wd}/montage')

    os.chdir(f'{_wd}/montage')

    a = f'{_wd}/InteractionEnergy/ie_.png'
    b = f'{_wd}/per_residue_matrix/matrix_.png'
    c = f'{_wd}/per_residue_receptor/heatmap_.png'
    d = f'{_wd}/per_residue_ligand/heatmap_.png'

    mount_panel(a, b, c, d, 'final_panel')



    if video == True:
        if not os.path.exists(f'{_wd}/montage/frames'):
            os.mkdir(f'{_wd}/montage/frames')
        
        os.chdir(f'{_wd}/montage/frames')

        frames = []

        digit_0 = len(str(time.frames[-1]))

        for t in time.frames:
            d_0 = len(str(t))
            add_zeros = '0' * (digit_0 - d_0)
            
            if not os.path.exists(f'frame_{add_zeros}{t}.png'):
                a = f'{_wd}/InteractionEnergy/ie_frames/ie_{t}.png'
                b = f'{_wd}/per_residue_matrix/matrix_frames/matrix_{t}.png'
                c = f'{_wd}/per_residue_receptor/heatmap_frames/heatmap_{t}.png'
                d = f'{_wd}/per_residue_ligand/heatmap_frames/heatmap_{t}.png'

                frames.append([a, b, c, d, f'frame_{add_zeros}{t}'])

        if frames != []:
            parallelizer.run(frames, mount_panel, n_procs, 'Mounting video frames')

        if not os.path.exists(f'{_wd}/montage/video.mp4_'):
            os.system(f'ffmpeg -i frame_%0{digit_0}d.png -vf format=yuv420p {_wd}/montage/video.mp4')

    os.chdir(_wd)



def geometry():
    run_rmsd()
    run_cmdist()
    run_rmsf()
    run_rgyr()
    run_ERMSD()
    final_mount()


if __name__ == '__main__':
    geometry()
    run_MMGBSA()
    run_ie_namd()
    run_res_ie()
    run_matrix_namd()
    panel_mounting()