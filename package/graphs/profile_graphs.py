import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress



def profile_graph(done_temp, avg_list, title, ylabel, name, colors, ylim=None, slope_start=0):
    c0 = colors[0]
    c1 = colors [1]

    #### this function plots average IFPcs vs temperature
    fig, axs = plt.subplots(nrows=1, ncols=1)

    first_last_t = [done_temp[0], done_temp[-1]]
    axs.scatter(done_temp, avg_list, c=c0)
    first_last_score = [slope_start, avg_list[-1]]

    try:
        f = np.poly1d(np.polyfit(first_last_t, first_last_score, 1))
        slope, intercept, r_value, p_value, std_err = linregress(first_last_t, first_last_score)
        axs.plot(done_temp, f(done_temp), color=c1, ls='--', label="MS = {:.5f}".format(slope))
        
    except Exception:
        slope = 'Impossible to calculate MS: required at least 2 TTMD steps'

    axs.set_title(title)
    axs.set_xlabel('Temperature (K)')
    axs.set_ylabel(ylabel)

    if ylim != None:
        bottom = ylim[0]
        top = ylim[1]

        if not bottom == None:
            axs.set_ylim(bottom=bottom)
        if not top == None:
            axs.set_ylim(top=top)

    axs.set_xlim(first_last_t)
    axs.legend()
    fig.savefig(f'{name}.png', dpi=300)

    return slope