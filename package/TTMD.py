import os
import sys
import glob
import importlib
import numpy as np
import multiprocessing
from  utilities import header
import parser.parser as parser


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
        self.resume = resume


    def run(self):
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


        REPLICA_class = REPLICA(VARS)
        for i in range(1, self.n_reps + 1):

            print(divider, f'\nRunning Replica {i}\n', divider)

            if not os.path.exists(f'RUN_{i}'):
                os.system(f'mkdir RUN_{i}')

            os.chdir(f'RUN_{i}')

            if not os.path.exists('__ENDED__'):

                REPLICA_class = REPLICA(VARS)

                REPLICA_class.prepare()

                REPLICA_class.equil1()

                REPLICA_class.equil2()

                REPLICA_class.calc_reference()

                REPLICA_class.simulation()
                
                REPLICA_class.graphs()

            os.chdir('..')



class REPLICA:
    def __init__(self, vars):
        self.__dict__ = vars
        self.check_trj_len = utils.check_trj_len(self.__dict__)
        # run utils with check_trj_len.[function]()


    def prepare(self):
        import simulation.system_preparation.system_prep as prep
        prepare_system = prep.system_preparation(self.__dict__)
        prepare_system.prepare()
        self.__dict__ |= prepare_system.__dict__
        wrapping = importlib.import_module('..wrapping', 'utilities.')
        self.wrapping = wrapping.wrapping(self.__dict__)


    def equil1(self):
        import simulation.equil.equil1 as eq1
        equil1 = eq1.equil1(self.__dict__)
        equil1.run()
        self.__dict__ |= equil1.__dict__


    def equil2(self):
        import simulation.equil.equil2 as eq2
        equil2 = eq2.equil2(self.__dict__)
        equil2.run()
        self.__dict__ |= equil2.__dict__


    def calc_reference(self):
        from scoring_function import run
        scoring = run.scoring(self.__dict__)
        scoring.run()
        self.__dict__ |= scoring.__dict__


    def simulation(self):
        from simulation import run
        simulation = run.simulation(self.__dict__)
        simulation.run()
        self.__dict__ |= simulation.__dict__


    def graphs(self):
        from graphs import run
        graphs = run.graphs(self.__dict__)
        graphs.draw()
        self.__dict__ |= graphs.__dict__


        


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


    from utilities import utils
    resume = utils.resume(VARS['device']).resume
    pid = utils.pid()
    
    RUN_class = RUN(VARS)
    RUN_class.run()
