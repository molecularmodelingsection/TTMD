import os
import importlib

utils = importlib.import_module('..utils', 'utilities.')


class equil1:
    def __init__(self, vars):
        self.__dict__ = vars

    def run(self):
        print('''\n——Running equil1''')
        if not os.path.exists('equil1'):
            os.mkdir('equil1')
        os.chdir('equil1')
        
        try:
            check = self.check_trj_len.check(self.solvprmtop, 'equil1.dcd', self.eq1len)
        except Exception:
            check = False


        if not os.path.exists('equil1.dcd') or check == False:
            with open("get_celldimension.vmd", 'w') as f:
                f.write(f"""mol delete all;
        mol load parm7 {self.solvprmtop} pdb {self.solvpdb}
        set all [atomselect top all];
        set box [measure minmax $all];
        set min [lindex $box 0];
        set max [lindex $box 1];
        set cell [vecsub $max $min];
        put "celldimension $cell"
        quit""")

            os.system(f"{self.vmd_path} -dispdev text -e get_celldimension.vmd > celldimension.log")

            with open("celldimension.log",'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('celldimension '):
                        dimension = str(line.split(' ')[1]) + ' ' + str(line.split(' ')[2]) + ' ' + str(line.split(' ')[3].rstrip('\n'))
                        
                        
            
            with open("equil1.nvt", 'w') as f:
                f.write(f"""parmfile {self.solvprmtop}
coordinates {self.solvpdb}
temperature {self.T_start}
timestep {self.timestep}
thermostat on
thermostatTemperature {self.T_start}
thermostatDamping 0.1
minimize {self.minsteps}
run {self.eq1len}ns
restart {self.resume}
PME on
cutoff 9.0
switching on
switchDistance 7.5
atomRestraint "nucleic or resname LIG" setpoints 5@0
trajectoryFile equil1.dcd
trajectoryPeriod {self.dcdfreq}
boxSize {dimension}""")

            os.system("rm get_celldimension.vmd celldimension.log")
            os.system(f"acemd3 --device {self.device} equil1.nvt")

        eq1out = {
            'coor': os.path.abspath('output.coor'),
            'vel': os.path.abspath('output.vel'),
            'xsc': os.path.abspath('output.xsc')
            }

        update = {'eq1': eq1out}

        self.output = {}
        self.output |= update

        os.chdir('..')
