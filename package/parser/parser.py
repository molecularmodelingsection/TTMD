import os
import sys
import glob
import ast
import argparse



class input_vars:
    def __init__(self):
        pass



    def parser(self):
        init = {}

        cmd = cmd_parser()
        config_file = cmd['config_file']
        file_vars = parse_config_file(config_file)

        for key in cmd:
            init[key] = cmd[key]

        if file_vars != {}:
            for key in file_vars:
                if not key in init.keys():
                    init[key] = file_vars[key]

        self.check_params(init)

        init = apply_defaults(self.__dict__)

        write_vars(self.__dict__)

        return self.__dict__



    def check_params(self, vars):
        keys = vars.keys()

        if vars['method'] == 'rt':
            self.method = 'rt'

        elif vars['method'] == 'ps':
            self.method = 'ps'

        else:
            try:
                m = vars['method']
            except Exception:
                m = None
            print(f'ERROR!\nMethod is \'{m}\'.\nSpecify a valid method [rt, ps]')
            sys.exit(0)



        ### RECEPTOR
        #check existence and correct format of receptor file
        if not 'receptor' in keys:
            print('Receptor path missing!\n(check your config_params)')
            sys.exit(0)

        try:
            self.receptor = os.path.abspath(vars['receptor'])

        except Exception:
            print('receptor path missing!\n(check your config_params)')
            sys.exit(0)

        if not os.path.exists(self.receptor):
            print(f'\'{self.receptor}\' is not a valid path')
            sys.exit(0)

        elif self.receptor[-3:] != 'pdb':
            print('Protein must be in pdb format')
            sys.exit(0)
            #check existence and correct format of protein file

        if self.method == 'ps':
            if 'receptor_resnum' in keys:
                self.receptor_resnum = vars['receptor_resnum']

        if self.method == 'ps':
            if 'receptor_shift' in keys:
                self.receptor_shift = vars['receptor_shift']



        ### LIGAND
        #check existence and correct format of ligand file
        if not 'ligand' in keys:
            print('Ligand path missing!\n(check your config_params)')
            sys.exit(0)

        try:
            self.ligand = os.path.abspath(vars['ligand'])

        except Exception:
            print('Ligand path missing!\n(check your config_params)')
            sys.exit(0)

        if not os.path.exists(self.ligand):
            print(f'{self.ligand} is not a valid path')
            sys.exit(0)

        if self.method == 'rt':
            if self.ligand[-4:] != 'mol2':
                print('Ligand must be in mol2 format')
                sys.exit(0)

            from rdkit import Chem
            from rdkit.Chem import AllChem

            #check existence of ligand charge
            mol = Chem.MolFromMol2File(self.ligand)
            self.ligand_charge = Chem.GetFormalCharge(mol)



        if self.method == 'ps':
            if self.ligand[-3:] != 'pdb':
                print('Ligand must be in pdb format')
                sys.exit(0)

            if 'ligand_resnum' in keys:
                self.ligand_resnum = vars['ligand_resnum']

            if 'ligand_shift' in keys:
                self.ligand_shift = vars['ligand_shift']

            if 'cutoff_dist' in keys:
                self.cutoff_dist = vars['cutoff_dist']
        

        if 'temp_ramp' in keys:
            temp_ramp = vars['temp_ramp']
            ramp_check = True
            for i,sublist in enumerate(temp_ramp):
                #check if the temperature step is correctly set
                t_start = sublist[0]
                t_end = sublist[1]
                T_step = sublist[2]
                if (t_end-t_start) % T_step != 0:
                    ramp_check = False
                    print('\nTemperature ramp is not set up correctly!')
                    print(f'--> List n° {i} contains an invalid temperature step ({T_step})\n')
                    
                #check if each list has the right number of elements
                num_el = len(sublist)
                if num_el != 4:
                    ramp_check = False
                    print('\nTemperature ramp is not set up correctly!')
                    print(f'--> List n° {i} contains only {num_el} elements!\n')

            #if one condition is not satisfied, exit the program
            if ramp_check == False:
                print(f'\nYour ramp: {temp_ramp}\nThe right way: [[T_start (K), T_end (K), T_step (K), step_len (ns)],]\n')
                sys.exit(0)

            self.temp_ramp = temp_ramp

        if 'score_stop' in keys:
            self.score_stop = vars['score_stop']

        if 'stop_range' in keys:
            self.stop_range = vars['stop_range']

        if 'padding'in keys:
            self.padding = vars['padding']

        if 'iso' in keys:
            self.iso = vars['iso']
        
        if 'timestep' in keys:
            self.timestep = vars['timestep']

        if 'dcdfreq' in keys:
            self.dcdfreq = vars['dcdfreq']

        if 'minsteps' in keys:
            self.minsteps = vars['minsteps']

        if 'eq1len' in keys:
            self.eq1len = vars['eq1len']

        if 'eq2len' in keys:
            self.eq2len = vars['eq2len']

        if 'dryer' in keys:
            self.dryer = vars['dryer']

        if 'smooth' in keys:
            self.smooth = vars['smooth']

        if 'device' in keys:
            self.device = vars['device']

        if 'n_procs' in keys:
            self.n_procs = vars['n_procs']

        if 'n_reps' in keys:
            self.n_reps = vars['n_reps']

        if 'vmd_path' in keys:
            #check if provided vmd path is correct: if not, search for local installation of vmd and use that instead
            vmd_check = True
            vmd_path = vars['vmd_path']
            #control first if vmd path is provided
            try:
                self.vmd_path = os.path.abspath(vmd_path)
            except Exception:
                print('\nVMD path missing! (check your config_params)\n')
                vmd_check = False

            #control if provided path is a valid path
            if vmd_check and not os.path.isfile(self.vmd_path):
                print(f'\n{vmd_path} is not a valid path\n')
                vmd_check = False
            #control if provided vmd path refers to a vmd installation
            if vmd_check and os.path.isfile(self.vmd_path):
                exe = self.vmd_path.split('/')[-1]
                if 'vmd' not in exe:
                    print(f'\n{self.vmd_path} is not a valid VMD executable\n')
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
                self.vmd_path = vmd_installed_path


        if self.method == 'ps':
            if 'namd_path' in keys:
                #check if provided namd path is correct: if not, search for local installation of namd and use that instead
                namd_check = True
                namd_path = vars['namd_path']
                #control first if namd path is provided
                try:
                    self.namd_path = os.path.abspath(namd_path)
                except Exception:
                    print('\nnamd path missing! (check your config_params)\n')
                    namd_check = False

                #control if provided path is a valid path
                if namd_check and not os.path.isfile(self.namd_path):
                    print(f'\n{namd_path} is not a valid path\n')
                    namd_check = False
                #control if provided namd path refers to a namd installation
                if namd_check and os.path.isfile(self.namd_path):
                    exe = self.namd_path.split('/')[-1]
                    if 'namd' not in exe:
                        print(f'\n{self.namd_path} is not a valid namd executable\n')
                        namd_check = False

                if not namd_check:
                    #if namd is not installed on local machine, exit from the program
                    import subprocess
                    try:
                        namd_installed_path = str(subprocess.check_output(['which','namd']))[2:-3]
                    except Exception:
                        print('\nnamd is not installed on your machine!\n')
                        sys.exit(0)
                    print(f'\nFound existing installation of namd at {namd_installed_path}')
                    print(f'Using {namd_installed_path}\n')
                    self.namd_path = namd_installed_path


        if 'rmsd_resids' in keys:
            if type(vars['rmsd_resids']) == str:
                try:
                    r = ast.literal_eval(vars['rmsd_resids'])
                    self.rmsd_resids = r

                except Exception:
                    sys.exit('Please check rmsd resids option')

            else:
                self.rmsd_resids = vars['rmsd_resids']

        if 'df' in keys:
            self.df = True

            
        return self.__dict__



def parse_config_file(config_file):
    conf = {}

    if config_file != None:
        if os.path.exists(config_file):
            
            with open(config_file, 'r') as config:
                lines = config.readlines()

                if lines != []:
                    for line in lines:
                        l = line.rstrip('\n')
                        if l != '' and [*l][0] != '#':
                            list = l.split('=')
                            key = list[0].strip(' ')
                            if len(list) == 2:
                                value = list[1].strip(' ')
                            else:
                                value = None

                            try:
                                x = ast.literal_eval(value)

                            except Exception:
                                x = value

                            conf[key] = x

                else:
                    print(f'empty {config_file}. Please specify a valid config_file')

        else:
            print(f'{config_file} does not exists')

    else:
        pass

    return conf



def cmd_parser():
    common_parser = argparse.ArgumentParser(description='ttmd', add_help=True, argument_default=argparse.SUPPRESS, conflict_handler='resolve')

    common_parser.add_argument("-f", "--config_file", type=str, help='Config file (command line options override config_file)', metavar='', dest='config_file', default=None)
    common_parser.add_argument('-m', '--method', help='choose rt or ps', metavar='', dest='method')
    common_parser.add_argument('-t', '--receptor', type=str, help='receptor file', metavar='', dest='receptor')
    common_parser.add_argument('-l', '--ligand', type=str, help='ligand file', metavar='', dest='ligand')

    ps = common_parser.add_argument_group('additional parameters for ps method')
    ps.add_argument('-tr', '--receptor_res', type=int, help='', metavar='', dest='receptor_resnum')
    ps.add_argument('-rshift', '--receptor_shift', type=int, help='', metavar='', dest='receptor_shift')
    ps.add_argument('-lr', '--ligand_res', type=int, help='', metavar='', dest='ligand_resnum')
    ps.add_argument('-lshift', '--ligand_shift', type=int, help='', metavar='', dest='ligand_shift')
    ps.add_argument('-co', '--cutoff', type=float, help='', metavar='', dest='cutoff_dist')
    ps.add_argument('-namd', '--namdpath', type=str, help='', metavar='', dest='namd_path')

    setup_group = common_parser.add_argument_group('TTMD Setup')
    setup_group.add_argument('-p', '--padding', type=float, help='', metavar='', dest='padding')
    setup_group.add_argument('-i', '--iso', help='', dest='iso')
    setup_group.add_argument('-temp', '--temp_ramp', type=list, help='', metavar='', dest='temp_ramp')
    setup_group.add_argument('-stop', '--score_stop', type=float, help='', metavar='', dest='score_stop')
    setup_group.add_argument('-range', '--stop_range', type=int, help='', metavar='', dest='stop_range')
    setup_group.add_argument('-ts', '--timestep', type=int, help='', metavar='', dest='timestep')
    setup_group.add_argument('-dcdf', '--dcdfreq', type=int, help='', metavar='', dest='dcdfreq')
    setup_group.add_argument('-min', '--minsteps', type=int, help='', metavar='', dest='minsteps')
    setup_group.add_argument('-eq1', '--eq1len', type=float, help='', metavar='', dest='eq1len')
    setup_group.add_argument('-eq2', '--eq2len', type=float, help='', metavar='', dest='eq2len')
    setup_group.add_argument('-dv', '--device', type=int, help='Index of GPU device to use for MD simulations, default=0', metavar='', dest='device')
    setup_group.add_argument('-np', '--n_procs', type=int, help='Number of CPU cores to use for trajectory analysis, default=4', metavar='', dest='n_procs')
    setup_group.add_argument('-vmd', '--vmdpath', type=str, help='', metavar='', dest='vmd_path', )
    setup_group.add_argument('-n', '--n_reps', type=int, help='', metavar='', dest='n_rep')
    setup_group.add_argument('-R', '--restart', default=False, action='store_true', help='', dest='restart')

    rmsd = common_parser.add_argument_group('Binding Site stability analysis')
    rmsd.add_argument('-df', '--denaturingfactor', default=False, action='store_true', help='', dest='df')
    rmsd.add_argument('-rmsd', '--rmsd', type=str, help='', metavar='', dest='rmsd_resids')
    
    args = vars(common_parser.parse_args())
    if args['df'] == False:
        args.pop('df')

    return args



def apply_defaults(dict):
    defaults = {
        'receptor_resnum': 25,
        'receptor_shift': 0,
        'ligand_resnum': 25,
        'ligand_shift': 0,
        'padding': 15,
        'iso': 'no',
        'temp_ramp': [[300,450,10,10]],
        'score_stop': 0.05,
        'stop_range': 10,
        'timestep': 2,
        'dcdfreq': 10000,
        'minsteps': 500,
        'eq1len': 0.1,
        'eq2len': 0.5,
        'dryer': 'yes',
        'smooth': 200,
        'n_procs': 4,
        'device': 0,
        'n_reps': 1,
        'df': False
    }

    for i in defaults:
        if not i in dict.keys():
            dict[i] = defaults[i]
        
    return dict



def write_vars(dict):
    with open('vars', 'w') as f:
        for k in dict:
            v = dict[k]
            f.write(f'{k} = {v}\n')