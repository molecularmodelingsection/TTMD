import os
import MDAnalysis as mda

class prepare:
    def __init__(self, dict):
        self.__dict__ = dict


    def complex_in(self, ions_rand):
        complexin = f'''source leaprc.protein.ff14SB
source leaprc.DNA.OL15
source leaprc.RNA.OL3
source leaprc.water.tip3p
source leaprc.gaff
loadoff atomic_ions.lib
loadamberparams frcmod.ionsjc_tip3p
REC = loadpdb {self.receptor}
COMPL = combine{{REC}}
saveAmberParm REC receptor.prmtop protein.inpcrd
saveAmberParm COMPL complex.prmtop complex.inpcrd
solvatebox COMPL TIP3PBOX {self.padding} {self._iso}
{ions_rand}
savepdb COMPL solv.pdb
saveamberparm COMPL solv.prmtop solv.inpcrd
quit'''

        with open("complex.in", 'w') as f:
            f.write(complexin)



    def prepare(self):
        file_exists = False
        try:
            with open('solv.pdb', 'r') as f:
                lines = f.readlines()
            if lines != []:
                file_exists = True
        except Exception:
            file_exists = False

        if file_exists == False:

            self.complex_in('')

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

            os.system("tleap -f complex.in")

            os.system(f"{self.vmd_path} -dispdev text -e determine_ions_fixed.vmd > ion.log")

            with open("ion.log",'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'Cations of charge 1' in line:
                        cations = str(line.split(':')[1].strip())
                    elif 'Anions of charge -1' in line:
                        anions = str(line.split(':')[1].strip())
                    else:
                        pass

            ions_rand = f'addIonsRand COMPL Na+ {cations} Cl- {anions} 5'

            self.complex_in(ions_rand)

            os.system(f"tleap -f complex.in")

            with open("check_charge.vmd", 'w') as f:
                f.write("""mol load parm7 solv.prmtop pdb solv.pdb
set all [atomselect top all]
set curr_charge [measure sumweights $all weight charge]
puts [format "\nCurrent system charge is: %.3f\n" $curr_charge]
exit""")

            os.system(f"{self.vmd_path} -dispdev text -e check_charge.vmd > charge.log")

            with open("charge.log",'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('Current system charge is: '):
                        end = str(line.split(':')[1].strip())
                if end.startswith("0.") or end.startswith("-0."):
                    pass
                else:
                    exit("Error: system charge is not 0!")

        receptor_top = os.path.abspath('receptor.prmtop')
        solvpdb = os.path.abspath('solv.pdb')
        solvprmtop = os.path.abspath('solv.prmtop')
        


        u = mda.Universe(self.receptor)
        sel = u.select_atoms('nucleic') #adodaro
        with mda.Writer('dry_protein.pdb', sel.n_atoms) as W:
            W.write(sel)

        dry_complex = f'''source leaprc.protein.ff14SB
source leaprc.DNA.OL15
source leaprc.RNA.OL3
source leaprc.gaff
loadoff atomic_ions.lib
PROT = loadpdb {self.receptor}
COMPL = combine{{PROT}}
saveAmberParm COMPL dry_complex.prmtop dry_complex.inpcrd
quit'''

        with open("dry_complex.in", 'w') as f:
            f.write(dry_complex)
 
        os.system("tleap -f dry_complex.in")

        complprmtop = os.path.abspath('dry_complex.prmtop')

        update = {
            'receptor_top': receptor_top,
            'complprmtop': complprmtop,
            'solvpdb': solvpdb,
            'solvprmtop': solvprmtop
            }

        return update
