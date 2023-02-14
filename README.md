# TTMD
Python code to run Thermal Titration Molecular Dynamics (TTMD) simulations

Reference publications:  
  1) **"Qualitative Estimation of Protein-Ligand Complex Stability through Thermal Titration Molecular Dynamics (TTMD) Simulations."**  
  Pavan M., Menin S., Bassani D., Sturlese M., Moro S. (published on *Journal of Chemical Information and Modeling*)  
  https://doi.org/10.1021/acs.jcim.2c00995  
  2) **"Thermal Titration Molecular Dynamics (TTMD): Not Your Usual Post-Docking Refinement"**  
  Menin S., Pavan M., Salmaso V., Sturlese M., Moro S. (published on *International Journal of Molecular Sciences*)  
  https://doi.org/10.3390/ijms24043596  

This script automatizes the passages needed to execute a TTMD simulation, from the system setup, to the equilibration protocol, the production phase and trajectory analyses. A TXT file is provided in order to reconstitute the right Python virtual environment needed to run the TTMD.py script. 
To reconstitute the right Python virtual environment to run the TTMD.py code:
- **conda create --name ttmd --file ttmd.txt**

The code relies on **external software dependency**:
- **Visual Molecular Dynamics (VMD)**  

System setup and parameterization for molecular dynamics is carried out by AmberTools22, which are installed within the provided conda environment. The current version of the script only supports the **ACEMD3 engine** to run molecular dynamics simulations. As for AmberTools22, ACEMD3 is also already installed within the provided conda environment.

To run a TTMD simulation on a protein-ligand complex of interest, create a folder containing the TTMD.py script plus the adequately prepared protein and ligand structures in the .pdb and .mol2 format respectively. **Editable settings can be provided either through the command line or through a configuration file (this option overrides command line arguments)**. 
To run the code:
1. open a terminal within the directory of interest
2. activate the right conda environment (**conda activate ttmd**)
3. run the code (**python3 TTMD.py [options]**)

To test the code, an example system is provided in the **test** directory.
