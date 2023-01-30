import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.stats import linregress


class graphs:
    def __init__(self, vars):
        self.__dict__ = vars

        self.denaturing_factor()



    def denaturing_factor(self):
        selection = ''
        for i,r in enumerate(self.rmsd_resids):
            if i == len(self.rmsd_resids) -1 :
                selection += f'resid {r}'
            else:
                selection += f'resid {r} or '

        ref_bonds = count_hbonds(self.solvprmtop, self.solvpdb, selection)

        mean_hbond = []

        for i in self.done_temp:
            hbonds = count_hbonds(self.solvprmtop, f'MD/swag_{i}.dcd', selection)
            avg = sum(hbonds) / len(hbonds)
            mean_hbond.append(avg)

        fig, axs = plt.subplots(nrows=1, ncols=1)
        
        temperature_list = []
        for set in self.temperature:
            temperature_list.append(set[0])

        first_last_t = [self.T_start, self.T_stop]
        axs.set_xlim(first_last_t)
        axs.set_ylim(-1,0)
        axs.scatter(temperature_list[:len(self.avg_list)], self.avg_list, c='royalblue')
        first_last_score = [-1.0, self.avg_list[-1]]

        f = np.poly1d(np.polyfit(first_last_t, first_last_score, 1))
        slope, intercept, r_value, p_value, std_err = linregress(first_last_t, first_last_score)
        axs.plot(temperature_list, f(temperature_list), color='tomato', ls='--', label="MS = {:.5f}".format(slope))
        axs.set_title('Titration Profile')
        axs.set_xlabel('Temperature (K)')
        axs.set_ylabel('Average IFP$_{CS}$')
        axs.set_ylim(-1,0)
        axs.set_xlim(first_last_t)
        axs.legend()
        fig.savefig('titration_profile.png', dpi=300)

        return slope





def count_hbonds(topology, trajectory, selection):
    u = mda.Universe(topology, trajectory)

    hbonds = HBA(universe=u)
    protein_hydrogens_sel = hbonds.guess_hydrogens(selection)
    protein_acceptors_sel = hbonds.guess_acceptors(selection)
    hbonds.hydrogens_sel = protein_hydrogens_sel
    hbonds.hydrogens_sel = protein_acceptors_sel

    hbonds.run()

    return hbonds.count_by_time()