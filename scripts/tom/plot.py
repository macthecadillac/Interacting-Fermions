import numpy as np
import matplotlib.pyplot as plt
import spinsys
import os
import re

font = {'size': 12}
plt.rc('font', **font)

Ns = [10, 12, 14, 16, 18]
nphi = 5000
ptype = 'var'           # plot type. entropy/agr/var
# colors = ['#002aff', '#008e00', '#ff7700', '#ff2600']

fig = plt.figure()
ax = fig.add_subplot(111)
# ax.tick_params(labelsize=8.5, length=3.5, width=0.5)
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax.spines[axis].set_linewidth(0.75)

for n, N in enumerate(Ns):
    try:
        plot_data = np.loadtxt('./N{}_{}.txt'.format(N, ptype))
    except FileNotFoundError:
        data_dir = './N{}'.format(N)
        flist = os.listdir(data_dir)
        regex = '{}[a-z0-9_.]+'.format(ptype)
        fs = [os.path.join(data_dir, f)
              for f in re.findall(regex, ' '.join(flist))]

        print('\n  Processing data for N = {}'.format(N))
        timer = spinsys.utils.timer.Timer(len(fs))
        arr = []
        for i, f in enumerate(fs):
            rawdata = np.loadtxt(f)
            if ptype is not 'agr':
                rawdata /= N
            h = float(re.search(r'[/a-zA-Z0-9]+_h([0-9.]+)[a-z0-9._]+',
                                f).groups()[0])
            dat = np.mean(rawdata)
            err = np.std(rawdata) / np.sqrt(nphi)
            arr.append((h, dat, err))
            timer.progress()

        arr.sort(key=lambda x: x[0])
        plot_data = np.array(arr)
        np.savetxt('./N{}_{}.txt'.format(N, ptype), plot_data)

    ax.errorbar(plot_data[:, 0], plot_data[:, 1], yerr=plot_data[:, 2],
                marker='o', markeredgecolor='black', markeredgewidth=0.9,
                markersize=5.5, linewidth=0.5, elinewidth=1, capsize=6,
                capthick=1.2, label='L = {}'.format(N))

plt.legend(frameon=False, prop={'size': 16})
plt.xlabel(r'$W$', fontsize=16)
ylabels = {'entropy': r'$S/L$',
           'agr': 'adjacent gap ratio',
           'var': r'$F/L$'}
plt.ylabel(ylabels[ptype], fontsize=16)
plt.tight_layout()
# plt.show()
plt.savefig('variance_plot.eps', dpi=200, orientation='landscape')
