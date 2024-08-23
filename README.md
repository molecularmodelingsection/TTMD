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
- `conda create --name ttmd --file env.txt`

The code relies on **external software dependency**:
- **Visual Molecular Dynamics (VMD)**  

System setup and parameterization for molecular dynamics is carried out by AmberTools22, which are installed within the provided conda environment. The current version of the script only supports the **ACEMD3 engine** to run molecular dynamics simulations. As for AmberTools22, ACEMD3 is already installed within the provided conda environment.

N.B. be sure to run the code with at least one Nvidia GPU device.

To run a TTMD simulation on a nucleic-peptide/protein complex of interest, create a folder containing the TTMD.py script plus the adequately prepared nucleic structure in the .pdb format and the peptide/protein in both .pdb and .mol2 format, naming it as 'ligand.extension'. **Editable settings can be provided either through the command line or a configuration file (this option overrides command line arguments)**. 
To run the code:
1. open a terminal within the directory of interest
2. activate the right conda environment (`conda activate ttmd`)
3. run the code (`python3 ttmd.py [options]`, detailed options can be displayed by executing `python3 ttmd.py --help`)

An example system is provided in the **test** directory to test the code.

Recent applications of the TTMD code:  

  **"PROTAC-Design-Evaluator (PRODE) : An Advanced Method for in-silico PROTAC design"**  
  A S Ben Geoffrey, Deepak Agrawal, Nagaraj M Kulkarni, Rajappan Vetrivel, Kishan Gurram  (published in *ACS Omega*)    
  https://doi.org/10.1021/acsomega.3c07318  
  **"A comprehensive study of SARS-CoV-2 main protease (Mpro) inhibitor-resistant mutants selected in a VSV-based system"**
  F Costacurta et al. (preprint in *BioRxiv*)  
  https://doi.org/10.1101/2023.09.22.558628  
  **"Structural Investigations on 2-Amidobenzimidazole Derivatives as New Inhibitors of Protein Kinase CK1 Delta"**
  S. Calenda et al. (published in *Pharmaceuticals*)
  https://doi.org/10.3390/ph17040468  
  **"A second life for the crystallographic structure of Berenil-dodecanucleotide complex: a computational revisitation thirty years after its publication"**  
  G. Novello et al. (preprint in *ResearchSquare*)  
  https://doi.org/10.21203/rs.3.rs-4269844/v1  
  **"Molecular Glue-Design-Evaluator (MOLDE): An Advanced Method for In-Silico Molecular Glue Design"**  
  A S Ben Geoffrey, Deepak Agrawal, Nagaraj M Kulkarni, G Manonmani (preprint in *Biorxiv*)  
  https://doi.org/10.1101/2024.08.06.606937
