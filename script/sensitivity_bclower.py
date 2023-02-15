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
# reformat the following to numpy array
# (b/c_lower, sensitivity, number of norms)
dat = np.array(
    [
    [1, 4, 128, 0],
    [1, 5, 128, 0],
    [2, 2.5, 128, 0],
    [2, 3, 128, 0],
    [2, 4, 192, 0],
    [2, 5, 192, 0],
    [2, 7, 384, 1],
    [2, 9, 384, 1],
    [3, 4, 256, 1],
    [3, 5, 256, 1],
    [3, 7, 384, 1],
    [3, 9, 384, 1]
    ])
dat
# %%
import matplotlib.cm as cm
import matplotlib
plt.clf()
fig, ax = plt.subplots()
colors = ['#084887', '#F58A07']
cmap = matplotlib.colors.ListedColormap(colors)
plt.scatter(dat[:, 1], dat[:, 0], s=dat[:, 2]/2, c=dat[:, 3]+1, cmap=cmap)
plt.xlabel(r'error sensitivity, $(1 - p_c^{\rm res \to res} )/ \mu$')
#plt.xlabel(r'$(1 - p_c^{\rm res \to res}) \times 10^3$')
#plt.xlabel('self-defection level ($\\times 10^{-3}$)')
plt.ylabel('$b/c$ lower bound')
plt.xlim(0, 10)
plt.ylim(0.0, 4)
plt.yticks([1, 2, 3])
arrow_opt = dict(width=0.12, alpha=0.6, fc='gray', ec='gray')
#arrow_opt = dict(head_width=0.3, head_length=0.2, width=0.12, fc='gray', ec='gray')
plt.arrow(7.0, 0.3, -1, 0, **arrow_opt)
plt.text(8.3, 0.3, 'more robust\nagainst errors', fontsize=12, horizontalalignment='center', verticalalignment='center')
arrow_opt = dict(width=0.18, head_length=0.3, alpha=0.6, fc='gray', ec='gray')
plt.arrow(0.5, 3.9, 0, -0.4, **arrow_opt)
plt.text(2.3, 3.7, 'more robust\nagainst mutants', fontsize=12, horizontalalignment='center', verticalalignment='center')
plt.text(2.8, 2.4, 'variants of L8', fontsize=14, verticalalignment='center', horizontalalignment='center', color=colors[0])
plt.text(3.95, 1.05, 'L8', fontsize=14, verticalalignment='bottom', horizontalalignment='right', color=colors[0])
plt.text(7.0, 3.4, 'variants of S16', fontsize=14, verticalalignment='center', horizontalalignment='center', color='#BF5700')
plt.text(6.92, 2.08, 'S16', fontsize=14, verticalalignment='bottom', horizontalalignment='right', color='#BF5700')
# plt.annotate('L8', xy=(4, 1), xytext=(2.5, 0.5), color=colors[0], fontsize=14, arrowprops=dict(facecolor='#333333', shrinkB=500, width=0.3, headwidth=3, headlength=5, connectionstyle='arc3,rad=-0.3'))
# plt.annotate('S16', xy=(7, 2), xytext=(7.0, 1.5), color='#BF5700', fontsize=14, arrowprops=dict(facecolor='#333333', shrink=0.2, width=0.3, headwidth=3, headlength=5, connectionstyle='arc3,rad=-0.3'))
plt.savefig('sensitivity_bclower.pdf')
plt.show()


# %%
