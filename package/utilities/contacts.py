import os
import pandas as pd
from collections import Counter
import MDAnalysis as mda



class resids:

    def __init__(self, vars):
        self.__dict__ = vars
        self.get_numeration()
        


    def get_numeration(self):
        self.receptor_dict = {}
        self.ligand_dict = {}

        rec_u = mda.Universe(self.receptor)
        rec = rec_u.residues
        self.receptor_len = len(rec)
        self.receptor_chain = [1, len(rec)]

        lig_u = mda.Universe(self.ligand)
        lig = lig_u.residues
        self.ligand_len = len(lig)
        self.ligand_chain = [len(rec) + 1, len(rec) + len(lig)]

        for i in range(1, self.receptor_len + 1):
            truenum = rec[i - 1].resnum + self.receptor_shift
            trueid = rec[i - 1].resname
            self.receptor_dict[i] = {
                                    'truenum': truenum,
                                    'trueid': trueid,
                                    }

        for i in range(1, self.ligand_len + 1):
            truenum = lig[i - 1].resnum + self.ligand_shift
            trueid = lig[i - 1].resname

            self.ligand_dict[i + self.receptor_len] = {
                                                        'truenum': truenum,
                                                        'trueid': trueid,
                                                        }



    def top_contacts(self, rec, lig):
        rec_res = list(self.__dict__[f'{rec}_dict'])
        rec_first = rec_res[0]
        rec_last = rec_res[-1]

        lig_res = list(self.__dict__[f'{lig}_dict'])
        lig_first = lig_res[0]
        lig_last = lig_res[-1]

        contact_sel = f"resnum {lig_first}:{lig_last} and same residue as around {self.cutoff_dist} resnum {rec_first}:{rec_last}"

        contactsList = []

        u = mda.Universe(self.solvpdb, self.output['eq2']['dcd'])

        ### iterate through each trajectory frame
        for ts in u.trajectory:
            #### create a ResidueGroup containing residues that are in contact with the ligand
            contacts = u.select_atoms(contact_sel).resids
            contactsResidsList = []

            for r in contacts:
                contactsResidsList.append(r)

            contactsList.extend(contactsResidsList)

        #### create a sorted list of all residues in contact with the associated number of contacts
        count_list = sorted(Counter(contactsList).items(), key = lambda x: x[1], reverse=True)

        if len(count_list) < self.__dict__[f'{lig}_resnum']:
            self.__dict__[f'{lig}_resnum'] = len(count_list)
        
        #### extract numResid best contacts and sort list
        top_list = [x[0] for x in count_list[:self.__dict__[f'{lig}_resnum']]]
        resnum_list = sorted(top_list, key=lambda x: int(x))
        resnames = []
        truenums = []

        #### extract resname from top contacts resids
        for r in resnum_list:
            dict = self.__dict__[f'{lig}_dict'][r]
            resnames.append(dict['trueid'])
            truenums.append(dict['truenum'])

        #### write contacts file
        with open(f'contacts_{lig}', 'w') as f:
            f.write('resnum,trueid,truenum\n')
            for i, id, truen in zip(resnum_list, resnames, truenums):
                f.write(f'{i},{id},{truen}\n')



    def residue_dict(self):

        dict = {
            'receptor': {},
            'ligand': {}
        }

        if not os.path.exists('contacts_ligand'):
            self.top_contacts('receptor', 'ligand')
        
        if not os.path.exists('contacts_receptor'):
            self.top_contacts('ligand', 'receptor')

        list = ['receptor', 'ligand']

        for f in list:
            d = pd.read_csv(f'contacts_{f}')

            r = d['resnum']
            id = d['trueid']
            n = d['truenum']

            for x,y,z in zip(r,id,n):

                dict[f][x] = {
                    'trueid': y,
                    'truenum':z
                    }

        print(dict)
        return dict