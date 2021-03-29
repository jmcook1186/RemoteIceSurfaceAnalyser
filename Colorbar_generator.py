import matplotlib as mpl
import matplotlib.pyplot as plt



def generate_colorbar(min_val, max_val, cmap, ticks, savepath):

    fig, ax = plt.subplots(1, 1)

    fraction = 1

    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cbar = ax.figure.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, pad=.05, extend='both', fraction=fraction,ticks=ticks)

    ax.axis('off')
    plt.savefig(savepath,dpi=300)
    
    return



# DEFINE OPTIONS

min_val = 0 # lowest value on colorbar
max_val = 20000 # highest value on colorbar
cmap = 'BuPu' # color scheme
ticks =[0, 5000, 10000, 15000, 20000] # labels to include on colorbar
savepath = '/home/joe/Code/Remote_Ice_Surface_Analyser/Manuscript/Figures/Colorbar_20000.jpg'
generate_colorbar(min_val, max_val, cmap, ticks, savepath)