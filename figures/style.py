import seaborn as sns
import matplotlib as mpl

def set_context(context):
    sns.set_context(context)
    sns.set_style('ticks')
    #sns.set_color_codes()


    sns.set_palette("Set1", 9, .75)


    # modifications to style

    mpl.rc('xtick')#, direction = 'in')
    mpl.rc('ytick')#, direction = 'in')

    #fontpath = '/usr/share/fonts/opentype/firacode/FiraCode-Regular.otf'
    #import matplotlib.font_manager
    #prop = matplotlib.font_manager.FontProperties(fname=fontpath)

    mpl.rcParams['font.sans-serif'] = ["Fira Sans"]
    mpl.rcParams['mathtext.default'] = 'regular'
    #mpl.rcParams['pdf.fonttype'] = 42  # illustrator fix
    mpl.rcParams['svg.fonttype'] = 'none'
    #mpl.rcParams['savefig.dpi'] = 200

def getColor(key):
    cdict = dict()
    cdict['d1'] = sns.color_palette()[0]
    cdict['a2a'] = sns.color_palette()[1]
    cdict['oprm1'] = sns.color_palette()[2]
    cdict['none'] = sns.color_palette()[8]
    cdict['sal'] = sns.color_palette()[3]
    cdict['fen'] = sns.color_palette()[2]
    #cdict['coc'] = sns.color_palette()[3]
    cdict['leftTurn'] = sns.color_palette()[2]
    cdict['rightTurn'] = sns.color_palette()[4]
    cdict['running'] = sns.color_palette()[0]
    cdict['stationary'] = sns.color_palette()[1]
    cdict['shuffled'] = 'gray'
    return cdict[key]

def lw():
    return 1.0