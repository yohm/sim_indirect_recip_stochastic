#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# %%
plt.rcParams['font.family'] ='sans-serif'
plt.rcParams["figure.subplot.left"] = 0.2
plt.rcParams["figure.subplot.right"] = 0.95
plt.rcParams["figure.subplot.bottom"] = 0.20
plt.rcParams["figure.subplot.top"] = 0.95
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['font.size'] = 16
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.facecolor'] = 'white'
#%%
c_levels = np.loadtxt('c_levels.dat')
mu = c_levels[:,0]
mu, c_levels.shape
# %%
# for each column in c_levels, plot mu, y
# log scale
plt.clf()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\mu$')
plt.ylabel(r'1 - $p_c^{\rm res \to res}$')
plt.plot(mu, mu, linestyle=(0, (5, 10)), linewidth=2)
for i in range(1, c_levels.shape[1]):
    y = 1.0 - c_levels[:,i]
    plt.plot(mu, y, '-', linewidth=0.3)
plt.show()
# %%

# get the data for mu=10^(-3)
mu = 10**(-3)
#c_levels = np.loadtxt('c_levels.dat')
mu_idx = np.where(c_levels[:,0] == mu)[0][0]
d_level_mu3 = 1 - c_levels[mu_idx, 1:]
#d_level_mu3 = np.round( (1 - c_levels[mu_idx, 1:]) * 1e4 )

# plot the histogram of c_level_mu3
plt.clf()
fig, ax = plt.subplots()
plt.hist(d_level_mu3 / mu, bins=51, color='gray', edgecolor='k', linewidth=0.5)
plt.xlabel(r'error sensitivity, $(1 - p_c^{\rm res \to res} )/ \mu$')
#plt.xlabel(r'$(1 - p_c^{\rm res \to res}) \times 10^3$')
#plt.xlabel('self-defection level ($\\times 10^{-3}$)')
plt.ylabel('# of norms')
plt.xlim(0, 10)
plt.yticks([0, 256, 512, 768])
#plt.text(0.5, 720, r'$\mu$ = 10$^{-3}$')
plt.arrow(2.0, 720, -1, 0, head_width=40, head_length=0.2, width=15, fc='gray', ec='gray')
plt.text(3.3, 720, 'more robust\nagainst errors', fontsize=12, horizontalalignment='center', verticalalignment='center')
plt.text(3.6, 500, 'L8', fontsize=14, verticalalignment='center', horizontalalignment='center')
plt.text(6.5, 700, 'S16', fontsize=14, verticalalignment='center', horizontalalignment='center')
plt.savefig('c_level_mu3_hist.pdf')
plt.show()

# %%
# find the index where c_level_mu3 is less than 0.0035
idx = np.where(d_level_mu3 < 0.0027)[0]
idx, len(idx), d_level_mu3[idx]
# %%

# %%
