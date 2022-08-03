# TTMD
Python code to run Thermal Titration Molecular Dynamics (TTMD) simulations

Reference publication:  
**"Qualitative Estimation of Protein-Ligand Complex Stability through Thermal Titration Molecular Dynamics (TTMD) Simulations."**  
Pavan M., Menin S., Bassani D., Sturlese M., Moro S. (under peer-review at *Journal of Chemical Information and Modeling*)

This script automatizes the passages needed to execute a TTMD simulation, from the system setup, to the equilibration protocol,
to the production phase and trajectory analyses. A YAML file is provided in order to reconstitute the right Python virtual 
environment needed to run the TTMD.py script. The code relies on **two external software dependencies**:
- **AMBER14**
- **Visual Molecular Dynamics (VMD)**  
The current version of the script only supports the **ACEMD3 engine** to run molecular dynamics simulations. ACEMD3 is already
installed within the provided conda environment.

To run a TTMD simulation on a protein-ligand complex of interest, create a folder containing the TTMD.py script plus the adequately
prepared protein and ligand structures in the .pdb and mol2 format respectively. **Editable settings can be changed at the beginning
of the script**. To run the code:
1. open a terminal within the directory of interest
2. activate the right conda environment (**conda activate ttmd**)
3. run the code (**python3 TTMD.py**)

To test the code, an example system is provided in **test** directory.
