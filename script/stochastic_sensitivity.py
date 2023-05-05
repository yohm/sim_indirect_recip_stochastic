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

# %%
def check_CESS(p1,p2,p3,p4):
    if p2 + p3 + p4 > 1.0 and p1 < 1.0 and p3 > 0.0:
        return True
    else:
        return False

#%%
def bc_lower(p1,p2,p3,p4):
    # b/c_lower = max { (p3+p4)/(1-p1), (p3+p4)/p3 }
    return max((p3+p4)/(1.0-p1), (p3+p4)/p3)

# %%
def sensitivity(p1,p2,p3,p4):
    chi = 1.0 / (p2 + p3 + p4 - 1.0)
    return 1.0 + (3.0 - p1) * chi

# %%
p2 = 1.0
all_dat = {}
for p1 in np.arange(0.0, 0.51, 0.1):
    y_list = []
    x_list = []
    for p3 in np.arange(1.0-p1, 1.01, 0.1):
        for p4 in np.arange(max(0,1.0-p1-p3+0.01), 1.01, 0.01):
            if check_CESS(p1,p2,p3,p4):
                print(p1, p3, p4, bc_lower(p1,p2,p3,p4), sensitivity(p1,p2,p3,p4))
                x_list.append(sensitivity(p1,p2,p3,p4))
                y_list.append(bc_lower(p1,p2,p3,p4))
            else:
                raise ValueError('invalid parameters')
    # sort x_list and y_list by x_list
    dat = np.array([x_list, y_list])
    dat = dat[:, dat[0].argsort()]
    # remove duplicate
    dat = dat[:, np.unique(dat[0], return_index=True)[1]]
    all_dat[p1] = dat

# %%

# iterate over all_dat
plt.clf()
for p1 in all_dat.keys():
    dat = all_dat[p1]
    plt.plot(dat[0], dat[1], label='p1={:.1f}'.format(p1))
plt.legend()

# %%
import matplotlib.cm as cm
import matplotlib
import matplotlib.patches as patches
plt.clf()
fig, ax = plt.subplots()
cmap = matplotlib.colors.ListedColormap('Blues')
plt.xlabel(r'error sensitivity, $(1 - p_c^{\rm res \to res} )/ \mu$')
plt.ylabel('$b/c$ lower bound')
plt.xlim(0, 6)
plt.ylim(0.0, 4)
plt.xticks([0,2,4,6])
plt.yticks([1, 2, 3])

arrow_opt = dict(width=0.05, head_length=0.07, alpha=0.6, fc='gray', ec='gray')
arrow_opt2 = dict(width=0.05, head_width=0.12, head_length=0.12, alpha=0.6, fc='gray', ec='gray')
ax.arrow(0.7, 0.1, -0.07, 0, transform=ax.transAxes, **arrow_opt)
ax.text(0.83, 0.1, 'more robust\nagainst errors', transform=ax.transAxes,
         fontsize=12, horizontalalignment='center', verticalalignment='center')
arrow_opt = dict(width=0.18, head_length=0.3, alpha=0.6, fc='gray', ec='gray')
ax.arrow(0.18, 0.82, 0, -0.1, transform=ax.transAxes, **arrow_opt2)
ax.text(0.18, 0.84, 'more robust\nagainst mutants', transform=ax.transAxes,
         fontsize=12, horizontalalignment='center', verticalalignment='bottom')

c = '#084887'
ax.text(3, 1.5, r'$p_1 = 0$', color=c, alpha=1.0,
        fontsize=12, horizontalalignment='right', verticalalignment='top')
ax.text(3.5, 2.0, r'$p_1 = 0.5$', color=c, alpha=0.5,
        fontsize=12, horizontalalignment='left', verticalalignment='bottom')
# ax.text(2.7, 3.5, r'larger $(p_3 + p_4)$',
#         fontsize=12, horizontalalignment='left', verticalalignment='bottom')
# ax.arrow(2.7, 3.3, -0.2, 0.5, head_width=0.1, head_length=0.1)
for (idx,p1) in enumerate(all_dat.keys()):
    dat = all_dat[p1]
    plt.plot(dat[0], dat[1], color=c, alpha=(1.0-idx/10.0), label='p1={:.1f}'.format(p1))

plt.savefig("stochastic_sensitivity.pdf")
# %%
