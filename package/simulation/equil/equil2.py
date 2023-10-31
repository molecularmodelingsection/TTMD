import os
import MDAnalysis as mda
import importlib



class equil2:
    def __init__(self, vars):
        self.__dict__ = vars
        
    def run(self):
        print('\n——Running equil2')
        if not os.path.exists('equil2'):
            os.mkdir('equil2')
        os.chdir('equil2')
        
        if os.path.exists('equil2.dcd'):
            try:
                check = self.check_trj_len.check(self.solvprmtop, 'equil2.dcd', self.eq2len)
            except Exception:
                check = False

        if not os.path.exists('equil2.dcd') or check == False:
            out = self.output['eq1']

            with open("equil2.npt", 'w') as f:
                f.write(f"""parmfile {self.solvprmtop}
coordinates {self.solvpdb}
binCoordinates {out['coor']}
binVelocities {out['vel']}
extendedSystem {out['xsc']}
temperature {self.T_start}
timestep {self.timestep}
thermostat on
thermostatTemperature {self.T_start}
thermostatDamping 0.1
barostat on
barostatPressure 1.01325
run {self.eq2len}ns
restart {self.resume}
PME on
cutoff 9.0
switching on
switchDistance 7.5
atomRestraint "nucleic and backbone or resname LIG" setpoints 5@0
trajectoryFile equil2.dcd
trajectoryPeriod {self.dcdfreq}""")

            os.system(f"acemd3 --device {self.device} equil2.npt")

        if os.path.exists('wrap.dcd'):
            try:
                check = self.check_trj_len.check(self.solvprmtop, 'wrap.dcd', self.eq2len)
            except Exception:
                check = False

        if not os.path.exists('wrap.dcd') or check == False:
            self.wrapping.wrap_equil2(self.solvpdb, 'equil2.dcd', 'wrap.dcd')

        if not os.path.exists('eq2_last.pdb'):
            wrap_u = mda.Universe(self.solvpdb, 'wrap.dcd')
            wrap_u.trajectory[-1]

            with mda.Writer('eq2_last.pdb', wrap_u.atoms.n_atoms) as W:
                W.write(wrap_u.atoms)

        eq2out = {
            'dcd': os.path.abspath('wrap.dcd'),
            'coor': os.path.abspath('output.coor'),
            'vel': os.path.abspath('output.vel'),
            'xsc': os.path.abspath('output.xsc')
            }

        update = {'eq2': eq2out}

        solvpdb = os.path.abspath('eq2_last.pdb')
        self.solvpdb = solvpdb

        self.output|= update

        os.chdir('..')
