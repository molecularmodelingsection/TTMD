#!/usr/bin/env python

import os
import sys
import glob
import importlib
import numpy as np
import multiprocessing
from utilities import header
from utilities import utils
from utilities import multiprocessing as mp
import parser.parser as parser
from replica import REPLICA
import time


np.set_printoptions(threshold=sys.maxsize)
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



def printdict(obj):
    for i in obj.__dict__:
        print(f'{i} = {obj.__dict__[i]}')

divider = '\n████████████████████████████████████████\n'


class RUN:
    def __init__(self, vars):
        self.__dict__ = vars


    def run(self, vars):
        self.temperature = []
        self.tot_len = 0

        for sublist in self.temp_ramp:
            t1 = sublist[0]
            t2 = sublist[1]
            tstep = sublist[2]
            length = sublist[3]
            

            for i in range(t1, t2 + tstep, tstep):
                self.temperature.append([i, length])
                self.tot_len += length
                
        self.T_start = self.temperature[0][0]
        self.T_stop = self.temperature[-1][0]

        if self.launch == 'serial':
            self.serial(vars)

        elif self.launch == 'parallel':
            self.parallel(vars)



    def serial(self, vars):
        for i in range(1, self.n_reps + 1):
            REPLICA_class = REPLICA(self.device, VARS)
            replica_run(i, self.device, vars)



    def parallel(self, vars):
        print(self.device)
        dv = {}
        for i,n in enumerate(self.device):
            dv[i+1] = n

        args = []
        for i in range(1, self.n_reps + 1):
            args.append([i, dv, vars])

        if __name__ == '__main__':
            parallelizer = mp.parallelizer(len(self.device))
            parallelizer.run(args, replica_run, 'running')



def replica_run(i, dv, vars):
    if vars['launch'] == 'parallel':
        proc = multiprocessing.current_process()._identity
        id = list(proc)[0]

        device = dv[id]

    elif vars['launch'] == 'serial':
        device = dv

    print(divider, f'\nRunning Replica {i}\n', divider)

    if not os.path.exists(f'RUN_{i}'):
        os.system(f'mkdir RUN_{i}')

    os.chdir(f'RUN_{i}')

    if not os.path.exists('__ENDED__'):
        print(f'replica {i}, device {device}')
        REPLICA_class = REPLICA(device, vars)

        REPLICA_class.prepare()

        REPLICA_class.equil1()

        REPLICA_class.equil2()

        REPLICA_class.calc_reference()

        REPLICA_class.simulation()

        ms, rmsd_slope, df_protein, df_prot_h2o = REPLICA_class.graphs()

        with open('__ENDED__', 'w') as f:
            f.write(f'''MS = {ms}
Binding Site RMSD slope = {rmsd_slope}

DF protein binding site = {df_protein}
DF binding site - water = {df_prot_h2o}\n\n\n''')

    os.chdir('..')


        


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")

    header.header()

    VARS = parser.input_vars().parser()

    print('\n** Parameters for your simulations were stored in vars.dat **\n')
    print('\n#######################################################\n')
    with open('vars.dat','w') as f:
        for i in VARS:
            line = f'{i} = {VARS[i]}'
            print(line)
            f.write(line + '\n')
    print('\n#######################################################\n')

    pid = utils.pid()
    
    RUN_class = RUN(VARS)
    RUN_class.run(VARS)
