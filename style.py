import seaborn as sns

def set_context():
    sns.set(context='paper', style='ticks', font_scale=.7, palette='tab10',
            rc={'font.size': 7,
                'axes.labelsize': 7,
                'axes.titlesize': 7,
                'xtick.labelsize': 6,
                'ytick.labelsize': 6,
                'legend.fontsize': 6,
                'axes.linewidth': .5,
                'axes.labelpad': 1.25,
                'axes.titlepad': 0,
                'lines.linewidth': .8,
                'lines.markersize': 3,
                'patch.linewidth': .5,
                'xtick.major.width': .5,
                'ytick.major.width': .5,
                'xtick.minor.width': .5,
                'ytick.minor.width': .5,
                'xtick.major.size': 2.5,
                'ytick.major.size': 2.5,
                'xtick.minor.size': 1.5,
                'ytick.minor.size': 1.5,
                'xtick.major.pad': 1.25,
                'ytick.major.pad': 1.25,
                'legend.handletextpad': .25,
                'legend.labelspacing': .25,
                'legend.handlelength': 1,
                'legend.borderaxespad': 0,
                'legend.borderpad': 0,
                'legend.frameon': False,
                'axes.facecolor': 'none',
                'axes.edgecolor': '0',
                'axes.grid': False,
                'axes.axisbelow': True,
                'axes.labelcolor': '0',
                'figure.facecolor': 'none',
                'text.color': '0',
                'xtick.color': '0',
                'ytick.color': '0',
                'patch.edgecolor': 'none',
                'font.family': ['sans-serif'],
                'font.sans-serif': ['Arial'],
                'mathtext.default': 'regular',
                'patch.force_edgecolor': False,
                'hatch.linewidth': .35,
                'svg.fonttype': 'none',
                'savefig.dpi': 300})
    
    
def getColor(key):
    cdict = dict()
    cdict['d1'] = sns.color_palette()[0] # or color by port -> cyan
    cdict['a2a'] = sns.color_palette()[3] # -> orange
    cdict['oprm1'] = sns.color_palette()[4]

    cdict['pL'] = sns.color_palette()[9]
    cdict['pL2C'] = cdict['pL']
    cdict['dL2C'] = cdict['pL']
#    cdict['pL2Cr'] = cdict['pL']
#    cdict['pL2Cd'] = tuple(list(cdict['pL']) + [.66])
#    cdict['pL2Co'] = tuple(list(cdict['pL']) + [.38])
    cdict['pR'] = sns.color_palette()[1]
    cdict['pR2C'] = cdict['pR']
    cdict['dR2C'] = cdict['pR']
#    cdict['pR2Cr'] = cdict['pR']
#    cdict['pR2Cd'] = tuple(list(cdict['pR']) + [.66])
#    cdict['pR2Co'] = tuple(list(cdict['pR']) + [.38])
    cdict['pC'] = sns.color_palette()[4]
    cdict['pC2L'] = cdict['pC']
    cdict['pC2R'] = cdict['pC']
#    cdict['pC2R'] = tuple(list(cdict['pC']) + [.38])
    
    cdict['mC2L'] = sns.color_palette()[0]
    cdict['mC2R'] = sns.color_palette()[3]
    cdict['mL2C'] = sns.color_palette()[2]
    cdict['mR2C'] = sns.color_palette()[5]
    
    cdict['correct'] = cdict['oprm1']
    cdict['error'] = sns.color_palette()[7]
    cdict['stay'] = cdict['d1']
    cdict['r.'] = cdict['stay']
    cdict['switch'] = cdict['a2a']
    cdict['o!'] = cdict['switch']
    cdict['o.'] = cdict['oprm1']
    
    cdict['double'] = sns.color_palette()[6]

    cdict['shuffled'] = (0,0,0)
    cdict['none'] = (0,0,0)
    
    #Open field
    cdict['stationary'] = sns.color_palette()[0]
    cdict['running'] = sns.color_palette()[1]
    cdict['leftTurn'] = sns.color_palette()[2]
    cdict['rightTurn'] = sns.color_palette()[3]
    
    #Decoding sorted by mutual information
    cdict['ascending'] = sns.color_palette()[0]
    cdict['descending'] = sns.color_palette()[1]
    cdict['random'] = sns.color_palette()[1]
    
    return cdict[key]

def lw():
    return 1.0