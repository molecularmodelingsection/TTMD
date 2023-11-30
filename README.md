# TTMD
Python code to run Thermal Titration Molecular Dynamics (TTMD) simulations

Reference publications:  
  1) **"Qualitative Estimation of Protein-Ligand Complex Stability through Thermal Titration Molecular Dynamics (TTMD) Simulations."**  
  Pavan M., Menin S., Bassani D., Sturlese M., Moro S. (published in *Journal of Chemical Information and Modeling*)  
  https://doi.org/10.1021/acs.jcim.2c00995  
  2) **"Thermal Titration Molecular Dynamics (TTMD): Not Your Usual Post-Docking Refinement"**  
  Menin S., Pavan M., Salmaso V., Sturlese M., Moro S. (published in *International Journal of Molecular Sciences*)    
  https://doi.org/10.3390/ijms24043596  
  4) **"Thermal Titration Molecular Dynamics (TTMD): Shedding Light on the Stability of RNA-Small Molecule Complexes"**  
  Dodaro A., Pavan M., Menin S., Salmaso V., Sturlese M., Moro S. (published in *Frontiers in Molecular Biosciences*)  
  https://doi.org/10.3389/fmolb.2023.1294543

This script automatizes the passages needed to execute a TTMD simulation, from the system setup to the equilibration protocol, the production phase, and trajectory analyses. A TXT file is provided in order to reconstitute the right Python virtual environment needed to run the TTMD.py script. 
To reconstitute the right Python virtual environment to run the TTMD.py code:

    conda create --name ttmd --file ttmd.txt

The code relies on **external software dependency**:
- **Visual Molecular Dynamics (VMD)**  

System setup and parameterization for molecular dynamics is carried out by AmberTools22, which are installed within the provided conda environment. The current version of the script only supports the **ACEMD3 engine** to run molecular dynamics simulations. As for AmberTools22, ACEMD3 is already installed within the provided conda environment.

To run a TTMD simulation on a protein-ligand complex of interest, create a folder containing the TTMD.py script plus the adequately prepared protein and ligand structures in the .pdb and .mol2 format respectively. **Editable settings can be provided either through the command line or a configuration file (this option overrides command line arguments)**. 
To run the code:
1. open a terminal within the directory of interest
2. activate the right conda environment (**conda activate ttmd**)
3. run the code (**python3 TTMD.py [options]**)

An example system is provided in the **test** directory to test the code.

This version uses ProLIF to calculate interaction fingerprints.

## **INSTALLATION NOTES**
- Use the provided conda env
- Install prolif v. 2.0.0 (maybe it needs to be installed by cloning it from github)
## **ISSUES**
This version works only for **nucleic acid-ligand complexes** residence time estimation.

(Do not use methods such as `ps` or `apo`).
