import importlib
from utilities import multiprocessing as mp
from utilities import utils



class REPLICA:
    def __init__(self, device, vars):
        self.__dict__ = vars
        self.device = device
        self.resume = utils.resume(self.device).resume
        self.parallelizer = mp.parallelizer(self.n_procs)
        self.check_trj_len = utils.check_trj_len(self.__dict__)

        palette_module = importlib.import_module('..palette', 'utilities.')
        self.colors = getattr(palette_module, vars['palette'])


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

        return self.ms, self.rmsd_slope, self.df_protein, self.df_prot_h2o
